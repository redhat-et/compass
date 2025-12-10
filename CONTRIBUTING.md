# Contributing to Compass

Thank you for your interest in contributing to Compass! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Communication](#communication)

## Code of Conduct

This project follows standard open source community guidelines. Be respectful, inclusive, and constructive in all interactions.

## Getting Started

### Setting Up Your Development Environment

1. **Fork the repository** to your own GitHub account
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/compass.git
   cd compass
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/redhat-et/compass.git
   ```
4. **Follow the Quick Start guide** in the [README.md](README.md#quick-start) to set up your environment

## Development Workflow

### 1. Discuss Before You Code

**For significant changes**, discuss your approach upfront:
- Open an issue describing the problem and proposed solution
- For substantial features or architectural changes, create a design document in `docs/`
- Wait for feedback from maintainers before investing significant effort
- Small bug fixes and typos don't require prior discussion

### 2. Work in Your Fork

- Develop all changes in your own fork
- Create feature branches from `main`:
  ```bash
  git checkout -b feature/your-feature-name
  ```
- Keep your fork's `main` branch in sync with upstream:
  ```bash
  git fetch upstream
  git checkout main
  git merge upstream/main
  ```

### 3. Keep PRs Small and Targeted

- **Aim for PRs under 500 lines** whenever possible (not counting auto-generated code)
- Each PR should address a single concern or feature
- Break large features into incremental PRs that preserve functionality
- Incremental PRs should:
  - Build on each other logically
  - Keep the codebase in a working state after each merge
  - Include tests for new functionality
  - Update documentation as needed

**Example of breaking up a large feature**:
```
PR 1: Add database schema for new feature (no breaking changes)
PR 2: Implement backend API endpoints with tests
PR 3: Add UI components
PR 4: Wire up UI to backend and add integration tests
```

### 4. Coordinate Breaking Changes

If a breaking change is unavoidable:
- Discuss in an issue first with the "breaking-change" label
- Document the migration path clearly
- Consider deprecation warnings before removal
- For very large refactors, consider working from a branch in the main repository
- Provide upgrade instructions in the PR description

## Pull Request Guidelines

### Before Submitting

1. **Rebase on upstream/main** to ensure clean integration:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
2. **Run tests locally**:
   ```bash
   make test
   make lint
   ```
3. **Update documentation** if your changes affect user-facing behavior
4. **Add tests** for new functionality or bug fixes

### PR Requirements

- **Title**: Clear, concise description of the change
  - Good: "Add support for INT8 quantization in model catalog"
  - Bad: "Fix stuff"
- **Description**: Include:
  - What changed and why
  - Link to related issue(s)
  - Testing performed
  - Breaking changes (if any)
  - Screenshots for UI changes
- **Size**: Keep PRs focused and reasonably sized
- **Tests**: Include unit tests, integration tests where appropriate
- **Documentation**: Update relevant docs in `docs/` and code comments

### PR Template

```markdown
## Description
[Brief description of changes]

## Related Issue
Fixes #[issue number]

## Changes Made
- [List key changes]

## Testing
- [ ] Unit tests pass (`make test-unit`)
- [ ] Integration tests pass (`make test-integration`)
- [ ] Manual testing performed

## Documentation
- [ ] Updated relevant documentation in `docs/`
- [ ] Updated code comments where needed
- [ ] Updated README if user-facing changes

## Breaking Changes
[Describe any breaking changes and migration path, or "None"]
```

### Review Process

- Maintainers will review PRs as time permits
- Address review feedback promptly
- Rebase and force-push if requested (don't create merge commits)
- Be open to suggestions and iterate on your implementation

## Commit Message Guidelines

### Format

Follow the [Conventional Commits](https://www.conventionalcommits.org/) style:

```
<type>: <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring (no functional changes)
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates

### Requirements

- **Subject line**: 50 characters or less, imperative mood
  - Good: "Add GPU cost calculation for H100"
  - Bad: "Added GPU costs" or "This adds GPU cost calculation"
- **Body**: Wrap at 72 characters, explain what and why (not how)
- **Sign-off**: Include DCO sign-off with `-s` flag:
  ```bash
  git commit -s -m "feat: Add support for custom SLO templates"
  ```
- **AI Assistance**: Nontrivial and substantial AI-generated or AI-assisted content should be “marked”.  Using the `Assisted-by:` tag is recommended as shown in the following example:

  ```
  Assisted-by: Claude <noreply@anthropic.com>
  ```

### Example Commit Message

```
feat: Add support for on-premise GPU cost models

Extends the cost calculation system to support user-provided pricing
for on-premise and existing GPU infrastructure. Previously only cloud
GPU rental pricing was supported.

Changes:
- Add cost_model field to DeploymentIntent schema
- Implement custom cost provider interface
- Update UI to accept user cost parameters

Related to #42

Signed-off-by: Your Name <your.email@example.com>
```

## Testing Requirements

### Test Coverage

- **Unit tests**: Required for all new functionality
  - Located in `tests/`
  - Run with `make test-unit`
- **Integration tests**: Required for API endpoints and workflows
  - Run with `make test-integration`
- **End-to-end tests**: For critical user flows (optional for most PRs)
  - Run with `make test-e2e`

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-e2e

# Run tests in watch mode during development
make test-watch

# Run linters
make lint

# Auto-format code
make format
```

### Writing Tests

- Test files should mirror source structure: `backend/src/foo/bar.py` → `tests/test_foo_bar.py`
- Use descriptive test names: `test_plan_capacity_with_minimum_accuracy_threshold()`
- Include both positive and negative test cases
- Mock external dependencies (databases, APIs, LLM calls)

## Documentation

### What to Document

- **User-facing changes**: Update `README.md` and relevant `docs/` files
- **API changes**: Update docstrings and API documentation
- **Architecture changes**: Update `docs/ARCHITECTURE.md`
- **New features**: Add usage examples and guides

### Documentation Style

- Use clear, concise language
- Include code examples where helpful
- Keep `CLAUDE.md` updated with project context for AI assistance
- Use Markdown formatting consistently

## Communication

### Stay in Touch

- **Rebase often**: Keep your branch up-to-date with upstream/main
  ```bash
  git fetch upstream
  git rebase upstream/main
  ```
- **Communicate frequently**:
  - Comment on issues and PRs
  - Ask questions if requirements are unclear
  - Update PRs if you're blocked or need help
- **Be responsive**: Address review feedback within a few days if possible

### Getting Help

- **Issues**: For bugs, feature requests, or questions
- **Discussions**: For general questions and brainstorming
- **Pull Requests**: For code review and technical discussion

## License

By contributing to Compass, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](LICENSE)).

## Questions?

If you have questions not covered here, please:
1. Check existing issues and documentation
2. Open a new issue with the "question" label
3. Reach out to maintainers

Thank you for contributing to Compass!
