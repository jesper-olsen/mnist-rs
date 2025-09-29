//! A simple crate for parsing the MNIST dataset.
//!
//! Provides utilities for loading image and label files, and includes
//! an optional feature for plotting images with gnuplot.

// Only bring in gnuplot if the "plotting" feature is enabled.
#[cfg(feature = "plotting")]
use gnuplot::{AxesCommon, Figure, Fix};

use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

pub mod error;
use error::MnistError;

const IMAGE_WIDTH: usize = 28;
const IMAGE_HEIGHT: usize = 28;
pub const NPIXELS: usize = IMAGE_WIDTH * IMAGE_HEIGHT;

/// Represents a single 28x28 MNIST image.
pub struct Image {
    pixels: [u8; NPIXELS], // row-major order
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

        let train_labels = read_labels(&dir.join("train-labels-idx1-ubyte"))?;
        let train_images = read_images(&dir.join("train-images-idx3-ubyte"))?;

        let test_labels = read_labels(&dir.join("t10k-labels-idx1-ubyte"))?;
        let test_images = read_images(&dir.join("t10k-images-idx3-ubyte"))?;

        assert!(train_labels.len()==train_labels.len());
        assert!(test_labels.len()==test_images.len());

        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }
}

