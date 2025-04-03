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

# Sequenzo: Fast, scalable, and intuitive social sequence analysis

Sequenzo is a high-performance Python package designed for social sequence analysis. It is built to analyze **any sequence of categorical events**, from individual career paths and migration patterns to corporate growth and urban development. 
Whether you are working with **people, places, or policies**, Sequenzo helps uncover meaningful patterns efficiently. 

Sequenzo outperforms traditional R-based tools in social sequence analysis, delivering faster processing and superior efficiency, especially for large-scale datasets. **No big data? No problem. You don’t need big data to benefit as Sequenzo is designed to enhance sequence analysis at any scale, making complex methods accessible to everyone.**

## Target Audience

Sequenzo is designed for:

- 🧑‍🎓 Quantitative researchers in sociology, demography, political science, economics, management, etc.
- 📊 Data scientists, data analysts, and business analysts working on trajectory/time-series clustering
- 🧑‍🏫 Educators teaching courses involving social sequence data
- 🔁 Users familiar with R packages like `TraMineR` who want a Python-native alternative


## Why Choose Sequenzo?

🚀 **High Performance**

Leverages Python’s computational power to achieve 10× faster processing than traditional R-based tools like TraMineR.

🎯 **Easy-to-Use API**

Designed with simplicity in mind: intuitive functions streamline complex sequence analysis without compromising flexibility.

🌍 **Flexible for Any Scenario**

Perfect for research, policy, and business, enabling seamless analysis of categorical data and its evolution over time.

## Documentation

Explore the full Sequenzo documentation [here](sequenzo.yuqi-liang.tech). Even though the documentation website is still under construction, you can already find some useful information there.

**Where to start on the documentation website?**
* New to Sequenzo or social sequence analysis? Begin with "About Sequenzo" → "Quickstart Guide" for a smooth introduction.
* Got your own data? After going through "About Sequenzo" and "Quickstart Guide", you are ready to dive in and start analyzing.
* Looking for more? Check out our example datasets and tutorials to deepen your understanding.

For Chinese users, additional tutorials are available on [Yuqi's video tutorials on Bilibili](https://space.bilibili.com/263594713/lists/4147974).

## Platform Compatibility

Sequenzo provides pre-built Python wheels for maximum compatibility:

| Platform        | Architecture       | Python Versions | Status     |
|-----------------|--------------------|------------------|------------|
| **macOS**       | universal2 (x86_64 + arm64) | 3.9, 3.10, 3.11 | Built-in wheel |
| **Windows**     | win32, AMD64       | 3.9, 3.10, 3.11 | Built-in wheel |
| **Linux (glibc)** | i686, x86_64      | 3.9, 3.10, 3.11 | `manylinux2014` |
| **Linux (musl)** | i686, x86_64      | 3.9, 3.10, 3.11 | `musllinux_1_2` |

## Installation

The latest stable release and required dependencies can be installed from [PyPI](https://pypi.org/project/sequenzo/). You can type the following line in your terminal:

```
pip install sequenzo
```

If you have some issues with the installation, it might because you have both Python 2 and Python 3 installed on your computer. In this case, you can try to use `pip3` instead of `pip` to install the package.

```
pip3 install sequenzo
```

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

## Team

**Authors**
* [Yuqi Liang, University of Oxford](https://www.yuqi-liang.tech/)
* [Xinyi Li, Heilongjiang University](https://github.com/Fantasy201)
* [Jan Heinrich Ernst Meyerhoff-Liang, Institute for New Economic Thinking Oxford](https://www.inet.ox.ac.uk/people/jan-meyerhoff-liang)

**Acknowledgements**
* Technical advisor in sequence analysis: [Professor Tim Liao, University of Illinois Urbana-Champaign](https://sociology.illinois.edu/directory/profile/tfliao)
* Website and related technical support: [Mactavish](https://github.com/mactavishz)
* Visual design consultant: Changyu Yi
* Sequence data sources compilation
  * Economics: Jan Meyerhoff-Liang
  * History: Jingrui Chen
* Yuqi's PhD advisor: [Professor Ridhi Kashyap, University of Oxford](https://www.nuffield.ox.ac.uk/people/profiles/ridhi-kashyap/)
