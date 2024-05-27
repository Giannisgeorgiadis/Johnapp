import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Upload data
st.title("Εφαρμογή Ανάλυσης Δεδομένων")
uploaded_file = st.file_uploader("Επιλέξτε ένα αρχείο CSV ή Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.write("Data Preview:")
    st.write(data.head())

    # 2D Visualization Tab
    tabs = st.tabs(["2D Visualization", "Classification", "Clustering", "Info"])

    with tabs[0]:
        st.subheader("2D Visualization")
        method = st.selectbox("Επιλέξτε μέθοδο μείωσης διάστασης", ["PCA", "t-SNE"])
        
        if method == "PCA":
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(data.iloc[:, :-1])
            plt.figure(figsize=(16, 10))
            sns.scatterplot(x=pca_results[:, 0], y=pca_results[:, 1], hue=data.iloc[:, -1])
            plt.xlabel('pca-one')
            plt.ylabel('pca-two')
            st.pyplot(plt)
        
        elif method == "t-SNE":
            tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
            tsne_results = tsne.fit_transform(data.iloc[:, :-1])
            plt.figure(figsize=(16, 10))
            sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=data.iloc[:, -1])
            plt.xlabel('tsne-one')
            plt.ylabel('tsne-two')
            st.pyplot(plt)

        # Exploratory Data Analysis (EDA)
        st.subheader("Exploratory Data Analysis (EDA)")
        st.write("Histogram of each feature:")
        for column in data.columns[:-1]:
            plt.figure(figsize=(10, 4))
            plt.hist(data[column], bins=30)
            plt.title(f'Histogram of {column}')
            st.pyplot(plt)

        st.write("Pairplot of the data:")
        sns.pairplot(data, hue=data.columns[-1])
        st.pyplot(plt)
