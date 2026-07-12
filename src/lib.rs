//! A simple crate for parsing the MNIST dataset.
//!
//! Provides utilities for loading image and label files, and includes
//! an optional feature for plotting images with gnuplot.

// Only bring in gnuplot if the "plotting" feature is enabled.
#[cfg(feature = "plotting")]
use gnuplot::{AxesCommon, Figure, Fix};

use rand::{Rng, RngExt};
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

pub mod error;
use error::MnistError;

pub const NUM_LABELS: usize = 10;
pub const IMAGE_WIDTH: usize = 28;
pub const IMAGE_HEIGHT: usize = 28;
pub const NPIXELS: usize = IMAGE_WIDTH * IMAGE_HEIGHT;

/// Represents a single 28x28 MNIST image.
#[derive(Clone)]
pub struct Image {
    pub pixels: [u8; NPIXELS], // row-major order
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const ASCII_GRADIENT: [char; 5] = [' ', '.', ':', '*', '@'];

        for i in 0..NPIXELS {
            let char_index = (self.pixels[i] as usize * ASCII_GRADIENT.len()) / 256;
            write!(f, "{}", ASCII_GRADIENT[char_index])?;
            if (i + 1) % IMAGE_WIDTH == 0 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

impl Image {
    /// Creates an image from a normalized f64 array.
    pub fn from_f64_array(fa: &[f64; NPIXELS]) -> Image {
        let mut pixels = [0u8; NPIXELS];
        fa.iter()
            .zip(pixels.iter_mut())
            .for_each(|(f, p)| *p = (f * 255.0) as u8);
        Image { pixels }
    }

    pub const fn width(&self) -> usize {
        IMAGE_WIDTH
    }

    pub const fn height(&self) -> usize {
        IMAGE_HEIGHT
    }

    /// Returns the raw pixel data as a slice of bytes.
    pub fn as_u8_array(&self) -> &[u8] {
        &self.pixels
    }

    /// Returns the pixel data as a 2D array.
    pub fn as_2d_array(&self) -> &[[u8; IMAGE_WIDTH]; IMAGE_HEIGHT] {
        // SAFETY: The memory layout of [u8; 784] is guaranteed to be identical
        // to [[u8; 28]; 28], so this transmutation is safe.
        unsafe { &*(self.pixels.as_ptr() as *const [[u8; IMAGE_WIDTH]; IMAGE_HEIGHT]) }
    }

    /// Returns the pixel data as a normalized f32 array (values 0.0 to 1.0).
    pub fn as_f32_array(&self) -> [f32; IMAGE_WIDTH * IMAGE_HEIGHT] {
        self.pixels.map(|p| p as f32 / 255.0)
    }

    /// Returns the pixel data as a normalized f64 array (values 0.0 to 1.0).
    pub fn as_f64_array(&self) -> [f64; NPIXELS] {
        self.pixels.map(|p| p as f64 / 255.0)
    }

    /// Returns a new Image shifted by the given amount in each direction.
    ///
    /// Positive `shift_x` moves content right, positive `shift_y` moves content down.
    /// Pixels shifted out of frame are dropped; vacated pixels are filled with 0 (black).
    /// This is the deterministic core used by both `random_shift` and `all_shifts`.
    pub fn shifted(&self, shift_x: i32, shift_y: i32) -> Self {
        let mut target_pixels = [0u8; NPIXELS];

        if shift_x == 0 && shift_y == 0 {
            target_pixels.copy_from_slice(&self.pixels);
            return Image {
                pixels: target_pixels,
            };
        }

        let w = IMAGE_WIDTH as i32;
        let h = (NPIXELS / IMAGE_WIDTH) as i32;

        let y_start = 0.max(-shift_y) as usize;
        let y_end = (h as usize).min((h - shift_y) as usize);
        let x_start = 0.max(-shift_x) as usize;
        let x_end = (w as usize).min((w - shift_x) as usize);
        let copy_width = x_end - x_start;

        for y in y_start..y_end {
            let src_y = (y as i32 + shift_y) as usize;
            let src_x = (x_start as i32 + shift_x) as usize;
            let dst_idx = y * IMAGE_WIDTH + x_start;
            let src_idx = src_y * IMAGE_WIDTH + src_x;

            target_pixels[dst_idx..dst_idx + copy_width]
                .copy_from_slice(&self.pixels[src_idx..src_idx + copy_width]);
        }

        Image {
            pixels: target_pixels,
        }
    }

    /// Returns a new Image shifted randomly by up to `max_shift` pixels in any direction.
    pub fn random_shift(&self, rng: &mut impl Rng, max_shift: i32) -> Self {
        let shift_x = rng.random_range(-max_shift..=max_shift);
        let shift_y = rng.random_range(-max_shift..=max_shift);
        self.shifted(shift_x, shift_y)
    }

    /// Returns every shift variant of this image for `shift_x, shift_y` each ranging
    /// over `-max_shift..=max_shift`, including the unshifted original (0, 0).
    ///
    /// For `max_shift = 1` this yields 9 images; for `max_shift = n` it yields
    /// `(2n + 1)^2` images. Useful when you want exhaustive augmentation rather
    /// than sampling a random subset via `random_shift`.
    pub fn all_shifts(&self, max_shift: i32) -> Vec<Image> {
        let side = (2 * max_shift + 1) as usize;
        let mut out = Vec::with_capacity(side * side);
        for shift_y in -max_shift..=max_shift {
            for shift_x in -max_shift..=max_shift {
                out.push(self.shifted(shift_x, shift_y));
            }
        }
        out
    }
}

fn read_u32(reader: &mut BufReader<File>) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

/// Reads the MNIST label file from the given path.
pub fn read_labels<P: AsRef<Path>>(path: P) -> Result<Vec<u8>, MnistError> {
    let file = File::open(path.as_ref())?;
    let mut reader = BufReader::new(file);

    let magic_number = read_u32(&mut reader)?;
    if magic_number != 2049 {
        return Err(MnistError::InvalidMagicNumber {
            expected: 2049,
            found: magic_number,
        });
    }

    let num_items = read_u32(&mut reader)?;
    let mut labels = vec![0u8; num_items as usize];
    reader.read_exact(&mut labels)?;

    Ok(labels)
}

/// Reads the MNIST image file from the given path.
pub fn read_images<P: AsRef<Path>>(path: P) -> Result<Vec<Image>, MnistError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let magic_number = read_u32(&mut reader)?;
    if magic_number != 2051 {
        return Err(MnistError::InvalidMagicNumber {
            expected: 2051,
            found: magic_number,
        });
    }

    let num_images = read_u32(&mut reader)?;
    let num_rows = read_u32(&mut reader)?;
    let num_cols = read_u32(&mut reader)?;

    if num_rows as usize != IMAGE_HEIGHT || num_cols as usize != IMAGE_WIDTH {
        return Err(MnistError::InvalidImageDimensions {
            expected: (IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32),
            found: (num_cols, num_rows),
        });
    }

    let mut images = Vec::with_capacity(num_images as usize);
    for _ in 0..num_images {
        let mut pixels = [0u8; NPIXELS];
        reader.read_exact(&mut pixels)?;
        images.push(Image { pixels });
    }

    Ok(images)
}

/// Plots an image using gnuplot.
///
/// This function is only available if the "plotting" feature is enabled.
#[cfg(feature = "plotting")]
pub fn plot(image: &Image, label: u8) {
    let mut fg = Figure::new();

    // gnuplot's image function expects data row by row, starting from the bottom-left.
    // Our data is top-to-bottom, so we reverse the rows.
    let z: Vec<u8> = image
        .pixels
        .chunks(IMAGE_WIDTH)
        .rev()
        .flatten()
        .copied()
        .collect();

    fg.axes2d()
        .set_aspect_ratio(Fix(1.0))
        .set_title(&format!("MNIST Label: {}", label), &[])
        .image(
            z.iter(),
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
            Some((0.0, 0.0, IMAGE_WIDTH as f64, IMAGE_HEIGHT as f64)),
            &[],
        );

    fg.show().unwrap();
}

pub fn flatten_image(image: &[[u8; 28]; 28]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(image.as_ptr() as *const u8, 28 * 28) }
}

pub fn unflatten_image(image: &[u8]) -> &[[u8; 28]; 28] {
    assert_eq!(image.len(), 28 * 28);
    unsafe { &*(image.as_ptr() as *const [[u8; 28]; 28]) }
}

pub struct Mnist {
    pub train_images: Vec<Image>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<Image>,
    pub test_labels: Vec<u8>,
}

impl Mnist {
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self, MnistError> {
        let dir = dir.as_ref();

        let train_labels = read_labels(dir.join("train-labels-idx1-ubyte"))?;
        let train_images = read_images(dir.join("train-images-idx3-ubyte"))?;

        let test_labels = read_labels(dir.join("t10k-labels-idx1-ubyte"))?;
        let test_images = read_images(dir.join("t10k-images-idx3-ubyte"))?;

        assert!(train_labels.len() == train_images.len());
        assert!(test_labels.len() == test_images.len());

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    /// Loads the dataset and exhaustively augments the training set in place with
    /// every shift variant in `-max_shift..=max_shift` (including the unshifted
    /// original). The test set is left untouched, since augmenting eval data would
    /// leak correlated variants across the train/test boundary.
    ///
    /// Training set size grows by a factor of `(2 * max_shift + 1)^2`, so for
    /// `max_shift = 1` (the sweet spot you found) that's a 9x increase.
    pub fn load_with_shift_augmentation<P: AsRef<Path>>(
        dir: P,
        max_shift: i32,
    ) -> Result<Self, MnistError> {
        let mut mnist = Self::load(dir)?;
        mnist.augment_train_with_shifts(max_shift);
        Ok(mnist)
    }

    /// Expands `train_images`/`train_labels` in place with every shift variant
    /// of each existing training image, `-max_shift..=max_shift` in both axes.
    pub fn augment_train_with_shifts(&mut self, max_shift: i32) {
        let side = (2 * max_shift + 1) as usize;
        let variants_per_image = side * side;
        let n = self.train_images.len();

        let mut new_images = Vec::with_capacity(n * variants_per_image);
        let mut new_labels = Vec::with_capacity(n * variants_per_image);

        for (image, &label) in self.train_images.iter().zip(self.train_labels.iter()) {
            for shifted in image.all_shifts(max_shift) {
                new_images.push(shifted);
                new_labels.push(label);
            }
        }

        self.train_images = new_images;
        self.train_labels = new_labels;
    }
}
