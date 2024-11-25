

# Streamlit Application Setup Guide

This guide walks you through setting up a Python virtual environment, installing the required dependencies, and running the Streamlit app.

---

## Prerequisites

Ensure the following are installed on your system:
- Python 3.7 or later
- pip (comes with Python 3.7+)

---

## Setup Instructions

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone <repository_url>
cd <repository_folder>
```
### 2. Create a Virtual Environment
Create a virtual environment named venv:

python3 -m venv venv
Activate the virtual environment:

macOS/Linux:
```bash
source venv/bin/activate
```
Windows:
```bash

venv\Scripts\activate
```
### 3. Install Dependencies
Install the required packages using requirements.txt:

```bash

pip install -r requirements.txt
```
### 4. Run the Streamlit Application
Run the Streamlit app using the following command:

```bash

streamlit run main.py
```
