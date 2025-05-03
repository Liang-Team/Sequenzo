<p align="center">
  <img src="https://raw.githubusercontent.com/Liang-Team/Sequenzo/main/assets/logo/FullLogo_NoBuffer.jpg" alt="Sequenzo Logo" width="400">
</p>

<p align="center">
  <!-- ✅ PyPI Latest Version Badge -->
  <a href="https://pypi.org/project/sequenzo/">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/sequenzo?color=blue">
  </a>

  <!-- 📦 Downloads Badge (可选) -->
  <a href="https://pypi.org/project/sequenzo/">
    <img alt="Downloads" src="https://static.pepy.tech/badge/sequenzo">
  </a>

  <!-- 📄 License Badge -->
  <a href="https://github.com/Liang-Team/Sequenzo/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/Liang-Team/Sequenzo">
  </a>
</p>

# Sequenzo: Fast, scalable, and intuitive social sequence analysis in Python

Sequenzo is a high-performance Python package designed for social sequence analysis. It is built to analyze **any sequence of categorical events**, from individual career paths and migration patterns to corporate growth and urban development. 
Whether you are working with **people, places, or policies**, Sequenzo helps uncover meaningful patterns efficiently. 

Sequenzo outperforms traditional R-based tools in social sequence analysis, delivering faster processing and superior efficiency, especially for large-scale datasets. **No big data? No problem. You don’t need big data to benefit as Sequenzo is designed to enhance sequence analysis at any scale, making complex methods accessible to everyone.**

## Target Users

Sequenzo is designed for:

- 🧑‍🎓 Quantitative researchers in sociology, demography, political science, economics, management, etc.
- 📊 Data scientists, data analysts, and business analysts working on trajectory/time-series clustering
- 🧑‍🏫 Educators teaching courses involving social sequence data
- 🔁 Users familiar with R packages such as `TraMineR` who want a Python-native alternative


## Why Choose Sequenzo?

🚀 **High Performance**

Leverages Python’s computational power to achieve 10× faster processing than traditional R-based tools like TraMineR.

🎯 **Easy-to-Use API**

Designed with simplicity in mind: intuitive functions streamline complex sequence analysis without compromising flexibility.

🌍 **Flexible for Any Scenario**

Perfect for research, policy, and business, enabling seamless analysis of categorical data and its evolution over time.

## Platform Compatibility

Sequenzo provides pre-built Python wheels for maximum compatibility — no need to compile from source.

| Platform         | Architecture                  | Python Versions      | Status            |
|------------------|-------------------------------|-----------------------|-------------------|
| **macOS**        | `universal2` (Intel + Apple Silicon) | 3.9, 3.11       | ✅ Pre-built wheel |
| **Windows**      | `AMD64` (64-bit)              | 3.9, 3.10, 3.11       | ✅ Pre-built wheel |
| **Linux (glibc)**| `x86_64` (standard Linux)     | 3.9, 3.10, 3.11       | ✅ Pre-built wheel |
| **Linux (musl)** | `x86_64` (Alpine Linux)       | 3.9, 3.10, 3.11       | ✅ Pre-built wheel |


What do these terms mean?
- **universal2 (macOS)**: One wheel supports both Intel (x86_64) and Apple Silicon (arm64) Macs.
- **manylinux2014 (glibc-based Linux)**: Compatible with most mainstream Linux distributions (e.g., Ubuntu, Debian, CentOS).
- **musllinux_1_2 (musl-based Linux)**: For lightweight Alpine Linux environments, common in Docker containers.
- **AMD64 (Windows)**: Standard 64-bit Windows system architecture.

All of these wheels are pre-built and available on PyPI — so `pip install sequenzo` should work on supported platforms, without needing a compiler.

**Windows (win32)** and **Linux (i686)** are dropped due to:

- Extremely low usage in modern systems (post-2020)
- Memory limitations (≤ 4GB) unsuitable for scientific computing workloads
- Increasing incompatibility with packages such as `numpy`, `scipy`, and `pybind11`
- Frequent build failures and maintenance overhead in CI/CD pipelines


## Installation

If you haven't installed Python, please follow [Yuqi's tutorial about how to set up Python and your virtual environment](https://www.yuqi-liang.tech/blog/setup-python-virtual-environment/). 

Once Python is installed, we highly recommend using [PyCharm](https://www.jetbrains.com/pycharm/download/) as your IDE (Integrated Development Environment — the place where you open your folder and files to work with Python), rather than Visual Studio. PyCharm has excellent built-in support for managing virtual environments, making your workflow much easier and more reliable.

In PyCharm, please make sure to select a virtual environment using Python 3.9, 3.10, or 3.11 as these versions are fully supported by `sequenzo`.

Then, you can open the built-in terminal by clicking the Terminal icon 
<img src="https://github.com/user-attachments/assets/1e9e3af0-4286-47ba-aa88-29c3288cb7cb" alt="terminal icon" width="30" style="display:inline; vertical-align:middle;"> 
in the left sidebar (usually near the bottom). It looks like a small command-line window icon.

Once it’s open, type the following to install `sequenzo`:

```
pip install sequenzo
```

If you have some issues with the installation, it might because you have both Python 2 and Python 3 installed on your computer. In this case, you can try to use `pip3` instead of `pip` to install the package.

```
pip3 install sequenzo
```

For more information about the latest stable release and required dependencies, please refer to [PyPI](https://pypi.org/project/sequenzo/). 

## Documentation

Explore the full Sequenzo documentation [here](sequenzo.yuqi-liang.tech). Even though the documentation website is still under construction, you can already find some useful information there.

**Where to start on the documentation website?**
* New to Sequenzo or social sequence analysis? Begin with "About Sequenzo" → "Quickstart Guide" for a smooth introduction.
* Got your own data? After going through "About Sequenzo" and "Quickstart Guide", you are ready to dive in and start analyzing.
* Looking for more? Check out our example datasets and tutorials to deepen your understanding.

For Chinese users, additional tutorials are available on [Yuqi's video tutorials on Bilibili](https://space.bilibili.com/263594713/lists/4147974).

## Join the Community

💬 **Have a question or found a bug?**

Please submit an issue on [GitHub Issues](https://github.com/Liang-Team/Sequenzo/issues). 

For an effective bug report, please include the following:
* A reproducible code example that clearly demonstrates the issue.
* The output you’re seeing, such as an error message or an image of the plot.
* A brief explanation of why you believe this is a bug.

Providing these details will help us diagnose and resolve the issue more efficiently. We are always happy to help and will address it as soon as possible.

🌟 **Enjoying Sequenzo?**

Support the project by starring ⭐ the GitHub repo and spreading the word!

🛠 **Interested in contributing?**

Check out our [contribution guide]() for more details (work in progress). 

* Write code? Submit a pull request to enhance Sequenzo.
* Testing? Try Sequenzo and share your feedback. Every suggestion counts!

If you're contributing or debugging, use:

```bash
pip install -r requirements-3.10.txt  # Or matching your Python version
```
    
For standard installation, use:

```bash
pip install .  # Uses pyproject.toml
```

## Team

**Authors**
* [Yuqi Liang, University of Oxford](https://www.yuqi-liang.tech/)
* [Xinyi Li, Heilongjiang University](https://github.com/Fantasy201)
* [Jan Heinrich Ernst Meyerhoff-Liang, Institute for New Economic Thinking Oxford](https://www.inet.ox.ac.uk/people/jan-meyerhoff-liang)

**Acknowledgements**
* Methodological advisor in sequence analysis: [Professor Tim Liao, University of Illinois Urbana-Champaign](https://sociology.illinois.edu/directory/profile/tfliao)
* With special thanks to our initial testers (alphabetically ordered): [Joji Chia](https://sociology.illinois.edu/directory/profile/jbchia2), [Kass Gonzalez](https://www.linkedin.com/in/kass-gonzalez-72a778276/), [Sinyee Lu](https://sociology.illinois.edu/directory/profile/qianyil4), [Sohee Shin](https://sociology.illinois.edu/directory/profile/sohees2)
* Website and related technical support: [Mactavish](https://github.com/mactavishz)
* Visual design consultant: Changyu Yi
* Sequence data sources compilation
  * Economics: Jan Meyerhoff-Liang
  * History: Jingrui Chen
* Yuqi's PhD advisor: [Professor Ridhi Kashyap, University of Oxford](https://www.nuffield.ox.ac.uk/people/profiles/ridhi-kashyap/)
* Yuqi's original programming mentor: [JiangHuShiNian](https://github.com/jianghushinian)

