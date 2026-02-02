# Contributing to RIPPLE

Thank you for your interest in contributing to RIPPLE! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please be kind and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/RIPPLE.git
   cd RIPPLE
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/RIPPLE.git
   ```

## Development Setup

### Prerequisites

- **Java 11+** (OpenJDK 17+ recommended)
- **Maven 3.8+**
- **Conda** (Miniconda or Anaconda)
- **Git**

### Setup Steps

1. **Create the conda environment:**
   ```bash
   conda env create -f conda/environment.yml
   # Or for CPU-only:
   conda env create -f conda/environment-cpu.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate ripple-env
   ```

3. **Build the Java application:**
   ```bash
   mvn clean package -DskipTests
   ```

4. **Run in development mode:**
   ```bash
   mvn exec:java
   ```

## Making Changes

1. **Create a new branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** following our [coding standards](#coding-standards)

3. **Test your changes** thoroughly

4. **Commit with clear messages:**
   ```bash
   git commit -m "feat: add new tracking algorithm"
   # or
   git commit -m "fix: resolve memory leak in video loader"
   ```

   We follow [Conventional Commits](https://www.conventionalcommits.org/) format:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `refactor:` - Code refactoring
   - `test:` - Adding/updating tests
   - `chore:` - Maintenance tasks

## Pull Request Process

1. **Update documentation** if you've changed functionality
2. **Ensure all tests pass** before submitting
3. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
4. **Open a Pull Request** against the `main` branch
5. **Fill out the PR template** with:
   - Description of changes
   - Related issue numbers
   - Screenshots (if UI changes)
   - Testing steps

## Coding Standards

### Java Code

- Follow [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html)
- Use meaningful variable and method names
- Add Javadoc comments for public methods
- Keep methods focused and under 50 lines when possible

### Python Code

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Use type hints for function parameters and return values
- Add docstrings for functions and classes
- Use `black` for formatting and `ruff` for linting

### General Guidelines

- Keep commits atomic and focused
- Write clear commit messages
- Remove debug code before committing
- Don't commit large binary files (use Git LFS if needed)

## Reporting Bugs

When reporting bugs, please include:

1. **Environment details:**
   - Operating system and version
   - Java version (`java -version`)
   - Python version (`python --version`)
   - GPU model (if applicable)

2. **Steps to reproduce** the issue

3. **Expected behavior** vs **actual behavior**

4. **Error messages** and stack traces

5. **Screenshots** if it's a UI issue

Use the GitHub Issues template for bug reports.

## Feature Requests

We welcome feature requests! Please:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** - why do you need this feature?
3. **Propose a solution** if you have one in mind
4. **Be patient** - we review requests regularly

## Questions?

If you have questions about contributing, feel free to:
- Open a GitHub Discussion
- Reach out to the maintainers

Thank you for helping make RIPPLE better! 🎉
