# Contributing to DIP-CS

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, PyTorch version)
- Error messages or logs

### Suggesting Features

Feature requests are welcome! Please:
- Check existing issues first
- Describe the feature and its use case
- Explain why it would be useful
- Provide examples if possible

### Code Contributions

#### Setup Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/Compressed-Sensing-with-Deep-Image-Prior-for-Chest-CT-Signal-Denoising.git
   cd Compressed-Sensing-with-Deep-Image-Prior-for-Chest-CT-Signal-Denoising
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

#### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code style guidelines

3. Test your changes:
   ```bash
   python examples/test_structure.py
   ```

4. Commit with clear messages:
   ```bash
   git commit -m "Add feature: description"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request

#### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Include type hints where appropriate
- Keep functions focused and modular

Example:
```python
def calculate_metric(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate similarity metric between two images.
    
    Args:
        image1: First image array
        image2: Second image array
        
    Returns:
        Similarity score as float
    """
    # Implementation
    pass
```

#### Documentation

- Update README.md if adding features
- Add docstrings to new functions/classes
- Update USAGE.md with examples
- Add comments for complex logic

#### Testing

While we don't have automated tests yet, please:
- Test your changes manually
- Verify examples still work
- Check imports are correct
- Test with different parameters

#### Pull Request Guidelines

Good PR should:
- Have a clear title and description
- Reference related issues
- Include only related changes
- Pass all checks
- Have clear commit messages

## Project Structure

Understanding the codebase:

```
src/
├── models/           # Neural network architectures
├── utils/           # Utility functions
│   ├── cs_utils.py      # Compressed sensing
│   ├── metrics.py       # Evaluation metrics
│   └── visualization.py # Plotting
└── reconstruction.py # Main reconstruction logic

examples/            # Usage examples
configs/            # Configuration files
```

## Areas for Contribution

We especially welcome contributions in:

### Features
- Additional network architectures
- More undersampling patterns
- Alternative optimization methods
- Support for 3D CT volumes
- Real-time reconstruction
- Multi-GPU support

### Documentation
- More examples
- Tutorials
- Better error messages
- Performance benchmarks

### Testing
- Unit tests
- Integration tests
- Performance tests
- Data validation

### Optimization
- Speed improvements
- Memory efficiency
- Better convergence
- Hyperparameter tuning

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion
- Contact the maintainers

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on what's best for the project
- Show empathy towards others

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉
