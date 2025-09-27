use clap::Parser;
use mnist::{plot, read_images, read_labels};
use std::path::PathBuf;

/// A demo application to showcase the mnist-parser library.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Path to the directory containing the MNIST dataset files.
    #[arg(short, long)]
    data_dir: PathBuf,

    /// Image number to show
    #[arg(short, long, default_value_t = 0)]
    image_number: usize,

    /// Train/Test set
    #[arg(long, default_value = "train")]
    dataset: String,

    /// Display the images using gnuplot (requires plotting feature).
    #[cfg(feature = "plotting")]
    #[arg(long)]
    plot: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Attempting to load MNIST data from: {:?}...", args.data_dir);

    let labels_path = args.data_dir.join("train-labels-idx1-ubyte");
    let images_path = args.data_dir.join("train-images-idx3-ubyte");
    let test_labels_path = args.data_dir.join("t10k-labels-idx1-ubyte");
    let test_images_path = args.data_dir.join("t10k-images-idx3-ubyte");
    let labels = read_labels(&labels_path)?;
    let images = read_images(&images_path)?;
    let test_labels = read_labels(&test_labels_path)?;
    let test_images = read_images(&test_images_path)?;

    println!("✓ Successfully loaded {} training labels.", labels.len());
    println!("✓ Successfully loaded {} training images.", images.len());
    println!("✓ Successfully loaded {} test labels.", test_labels.len());
    println!("✓ Successfully loaded {} test images.", test_images.len());

    let pixels = images[0].as_u8_array();
    assert_eq!(pixels.len(), 784);
    assert_eq!(labels.len(), 60000);
    assert_eq!(images.len(), 60000);
    assert_eq!(test_labels.len(), 10000);
    assert_eq!(test_images.len(), 10000);

    let (images, labels) = match args.dataset.as_str() {
        "train" => (images, labels),
        "test" => (test_images, test_labels),
        _ => return Err("expected train or test".into()),
    };

    if args.image_number >= images.len() {
        return Err("Image number out of range".into());
    }

    println!(
        "\n--- Dataset: {} | Image #{} | Label: {} ---",
        args.dataset, args.image_number, labels[args.image_number]
    );
    println!("{}", images[args.image_number]);

    // This entire block is compiled out if the "plotting" feature is not enabled.
    #[cfg(feature = "plotting")]
    {
        if args.plot {
            plot(&images[args.image_number], labels[args.image_number]);
        } else {
            println!("\nHint: Re-run with the --plot flag to see graphical plots.");
        }
    }

    println!("\nDemo finished successfully!");
    Ok(())
}
