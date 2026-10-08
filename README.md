# Brain Tumor MRI Classification Streamlit App

This repository hosts a simple and clean Streamlit application for brain tumor classification using MRI scans. It is powered by a pre-trained **ConvNeXtTiny** deep learning model trained to identify 4 target categories: Glioma, Meningioma, Pituitary, and No Tumor.

## 🚀 How to Run Locally

### 1. Prerequisiutes & Installation
Clone this repository and navigate to the application folder:
```bash
cd brain_tumor_app
```
Install all necessary dependencies:
```bash
pip install -r requirements.txt
```

### 2. Model Weight Placement
Since the pre-trained `convnexttiny_final.h5` model size (~281 MB) exceeds standard GitHub file limits (100 MB), you have three options:
* **Local Copy**: Manually copy the compiled checkpoint from Google Drive (`/content/drive/MyDrive/models/convnexttiny_final.h5`) directly into this application root folder under the name `convnexttiny_final.h5`.
* **Hugging Face Hub hosting**: Store the model on Hugging Face Hub, copy the direct download link, and paste it into the **Model Download URL** textbox in the app sidebar.
* **Git LFS**: Track the model with Git Large File Storage before pushing to GitHub:
  ```bash
  git lfs install
  git lfs track "*.h5"
  git add .gitattributes
  ```

### 3. Launch App
Execute the Streamlit launch command:
```bash
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Community Cloud

1. Push your code files (with `.gitattributes` if using Git LFS, or exclude `.h5` files via `.gitignore`) to a public GitHub repository.
2. Navigate to [Streamlit Community Cloud](https://share.streamlit.io/) and authorize access to your GitHub account.
3. Create a new app pointing to your repository, branch, and specify the Entry Point path as `app.py`.
4. If you chose not to upload the model weight file directly via Git LFS, use the sidebar text box to download the model on startup via URL.
