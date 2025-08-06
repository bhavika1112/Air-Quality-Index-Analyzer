import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# App Configuration
st.set_page_config(page_title="AQI Analyzer", layout="wide")
st.title("🌍 Air Quality Index (AQI) Prediction & Analysis")

# File Uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df.dropna()

if uploaded_file:
    df = load_data(uploaded_file)
    st.session_state.df = df
    
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Initialize all tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "🤖 Model Training", "🔮 Prediction", "📚 About"])

    # -------------------------
    # Visualization Tab
    # -------------------------
    with tab1:
        st.header("Data Visualization")
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y %H:%M')
            df = df.set_index('date')
        numeric_df = df.select_dtypes(include=['float64', 'int64'])

        plot_type = st.selectbox("Select visualization type:", 
                               ["Histogram", "Bar Graph", "Boxplot", 
                                "Subplots", "Scatter Plot"])

        selected_feature = st.selectbox("Select primary feature:", numeric_df.columns)
        
        secondary_feature = None
        if plot_type in ["Bar Graph", "Boxplot", "Scatter Plot"]:
            secondary_feature = st.selectbox("Select secondary feature:", 
                                           numeric_df.columns.drop(selected_feature))

        plt.figure(figsize=(10, 6))
        
        if plot_type == "Histogram":
            sns.histplot(numeric_df[selected_feature], kde=True, bins=20)
            plt.title(f"Distribution of {selected_feature}")
            
        elif plot_type == "Boxplot":
            sns.boxplot(data=numeric_df[selected_feature])
            plt.title(f"Boxplot of {selected_feature}")
            
        elif plot_type == "Bar Graph":
            if 'date' in df.columns:
                # Time-based bar graph with proper Streamlit rendering
                frequency = st.selectbox("Select time frequency:", ['Daily', 'Weekly', 'Monthly'])
                freq_map = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M'}
                
                # Ensure datetime index and numeric data
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                numeric_df = df.select_dtypes(include=[np.number])
                
                # Resample with error handling
                resampled_df = numeric_df[selected_feature].resample(freq_map[frequency]).mean()
                if resampled_df.empty:
                    st.error("No data to plot after resampling!")
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(resampled_df.index.astype(str), resampled_df.values)
                    plt.title(f"{frequency} Average: {selected_feature}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()

            else:
                # Simplified bar graph for non-date data
                if selected_feature not in numeric_df.columns:
                    st.error(f"Column '{selected_feature}' not found!")
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(numeric_df.index.astype(str), numeric_df[selected_feature])
                    plt.title(f"Distribution of {selected_feature}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    plt.close()
                
        elif plot_type == "Subplots":
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for i, col in enumerate(numeric_df.columns[:4]):
                sns.lineplot(data=numeric_df[col], ax=axes[i//2, i%2])
                axes[i//2, i%2].set_title(col)
            plt.tight_layout()
            
        elif plot_type == "Scatter Plot":
            sns.scatterplot(data=numeric_df, 
                          x=selected_feature, 
                          y=secondary_feature)
            plt.title(f"{selected_feature} vs {secondary_feature}")
            
        st.pyplot(plt)

        # Correlation Heatmap
        st.write("### Correlation Matrix")
        corr_matrix = numeric_df.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(plt)

    # -------------------------
    # Model Training Tab
    # -------------------------
    with tab2:
        st.header("Machine Learning Models")
        
        features = st.multiselect("Select features:", df.columns)
        
        if features:
            model_name = st.selectbox("Select model:", 
                ["Linear Regression", "KNN", "Decision Tree", 
                 "K-Means Clustering", "Agglomerative Clustering"]
            )

            if model_name in ["Linear Regression", "KNN", "Decision Tree"]:
                target = st.selectbox("Select target variable:", df.columns)
                
                test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                if model_name == "Linear Regression":
                    model = LinearRegression()
                elif model_name == "KNN":
                    model = KNeighborsRegressor()
                elif model_name == "Decision Tree":
                    model = DecisionTreeRegressor()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.write("### Model Performance")
                st.write(f"**R² Score**: {r2_score(y_test, y_pred):.2f}")
                st.write(f"**MAE**: {mean_absolute_error(y_test, y_pred):.2f}")
                st.write(f"**RMSE**: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

                plt.figure(figsize=(8, 4))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title("Actual vs Predicted")
                st.pyplot(plt)

            else:
                # CLUSTERING MODELS
                st.subheader("Clustering Results")
                X_cluster = df[features].values
                X_scaled = StandardScaler().fit_transform(X_cluster)
                
                if model_name == "K-Means Clustering":
                    n_clusters = st.slider("Number of clusters:", 2, 5, 3)
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = model.fit_predict(X_scaled)
                    
                    # K-Means metrics
                    st.write("### Clustering Performance")
                    st.write(f"**Inertia**: {model.inertia_:.2f}")
                    st.write("*Lower values indicate tighter clusters*")
                    
                elif model_name == "Agglomerative Clustering":
                    n_clusters = st.slider("Number of clusters:", 2, 5, 3)
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                    clusters = model.fit_predict(X_scaled)
                    
                    # Agglomerative metrics
                    from sklearn.metrics import silhouette_score
                    silhouette = silhouette_score(X_scaled, clusters)
                    st.write("### Clustering Performance")
                    st.write(f"**Silhouette Score**: {silhouette:.2f}")
                    st.write("*Closer to 1 means better defined clusters*")
                
                # Visualization
                plt.figure(figsize=(10, 6))
                if len(features) >= 2:
                    plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=clusters, cmap='viridis', alpha=0.6)
                    plt.xlabel(features[0])
                    plt.ylabel(features[1])
                else:
                    plt.hist([X_cluster[clusters == i] for i in range(n_clusters)], 
                             stacked=True, label=[f"Cluster {i}" for i in range(n_clusters)])
                    plt.xlabel(features[0])
                    plt.legend()
                
                plt.title(f"{model_name} Results")
                st.pyplot(plt)
                
                # Cluster statistics
                df['Cluster'] = clusters
                st.write("### Cluster Distribution")
                st.bar_chart(df['Cluster'].value_counts())
                
                st.write("### Cluster Averages")
                st.dataframe(df.groupby('Cluster')[features].mean())

    # -------------------------
    # Prediction Tab (Restored)
    # -------------------------
    with tab3:
        st.header("AQI Prediction")
        
        if 'model' in locals() and 'features' in locals():
            user_inputs = {}
            cols = st.columns(2)
            for i, feature in enumerate(features):
                user_inputs[feature] = cols[i % 2].number_input(
                    f"Enter {feature}", 
                    value=float(df[feature].mean()),
                    step=0.1
                )
            
            if st.button("Predict"):
                input_df = pd.DataFrame([user_inputs])
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted {target}: **{prediction:.2f}**")
        else:
            st.info("Please train a model first in the Model Training tab")

    # -------------------------
    # About Tab
    # -------------------------
    with tab4:
        st.header("About")
        st.markdown("""
        **Air Quality Analysis Tool**:
        - Visualize pollution data
        - Train machine learning models
        - Make predictions
        - Discover patterns with clustering
        """)