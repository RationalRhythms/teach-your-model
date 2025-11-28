from setuptools import setup, find_packages

setup(
    name="teach-your-model",
    version="1.0.0",
    description="Teach your AI model through interactive feedback - it learns from your corrections",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': [
            'models/production/model.joblib',
            'data/labeled_pool/initial_labeled.csv'
        ],
    },
    entry_points={
        'console_scripts': [
            'tmodel=cli:cli',
        ],
    },
    python_requires=">=3.8",
)