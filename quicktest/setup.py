from setuptools import find_packages, setup

# pip install cleverhans['base, tf']
setup(
    name="src",
    extras_require=dict(
        base=[
            "nose",
            "pycodestyle",
            "scipy",
            "matplotlib",
            "mnist",
            "numpy",
            "tensorflow-probability",
            "joblib",
            "easydict",
            "absl-py",
            "six",
        ],
        tf=[
            "tensorflow>=2.4.0",
            "tensorflow-probability",
            "tensorflow-datasets",
        ],
        pytorch=[
            "torch>=1.7.0",
            "torchvision>=0.8.0",
        ],
    ),
    packages=find_packages(),
)
