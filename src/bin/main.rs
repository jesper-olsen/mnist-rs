use clap::Parser;
use mnist::{Mnist, plot};
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

    let data = Mnist::load(args.data_dir)?;
    println!(
        "✓ Successfully loaded {} training labels.",
        data.train_labels.len()
    );
    println!(
        "✓ Successfully loaded {} training images.",
        data.train_images.len()
    );
    println!(
        "✓ Successfully loaded {} test labels.",
        data.test_labels.len()
    );
    println!(
        "✓ Successfully loaded {} test images.",
        data.test_images.len()
    );

    let pixels = data.train_images[0].as_u8_array();
    assert_eq!(pixels.len(), 784);
    assert_eq!(data.train_labels.len(), 60000);
    assert_eq!(data.train_images.len(), 60000);
    assert_eq!(data.test_labels.len(), 10000);
    assert_eq!(data.test_images.len(), 10000);

    let (images, labels) = match args.dataset.as_str() {
        "train" => (data.train_images, data.train_labels),
        "test" => (data.test_images, data.test_labels),
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
