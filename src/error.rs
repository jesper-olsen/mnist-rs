use crate::fmt;
use std::error::Error;
use std::io;

#[derive(Debug)]
pub enum MnistError {
    /// An error occurred during file I/O (e.g., file not found, permission denied).
    /// This wraps the underlying `std::io::Error`.
    Io(io::Error),

    /// The file's magic number was incorrect, indicating a corrupt or wrong file type.
    InvalidMagicNumber { expected: u32, found: u32 },

    /// The image dimensions in the file header do not match the expected 28x28.
    InvalidImageDimensions {
        expected: (u32, u32),
        found: (u32, u32),
    },
}

impl fmt::Display for MnistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MnistError::Io(e) => write!(f, "I/O error: {e}"),
            MnistError::InvalidMagicNumber { expected, found } => write!(
                f,
                "Invalid magic number. Expected {expected}, but found {found}"
            ),
            MnistError::InvalidImageDimensions { expected, found } => write!(
                f,
                "Invalid image dimensions. Expected {}x{}, but found {}x{}",
                expected.0, expected.1, found.0, found.1
            ),
        }
    }
}

impl Error for MnistError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            MnistError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for MnistError {
    fn from(err: io::Error) -> Self {
        MnistError::Io(err)
    }
}
