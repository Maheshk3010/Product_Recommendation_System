import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Users\sanum\Desktop\recommendation_system\ratings.csv',
                     header=None, names=['userId', 'productId', 'rating', 'timestamp'])
    df = df.drop('timestamp', axis=1)
    
    user_counts = df.groupby('userId')['rating'].count()
    filtered_users = user_counts[user_counts >= 50].index
    df_filtered = df[df['userId'].isin(filtered_users)]
    
    product_counts = df_filtered.groupby('productId')['rating'].count()
    filtered_products = product_counts[product_counts >= 10].index
    df_filtered = df_filtered[df_filtered['productId'].isin(filtered_products)]
    
    return df_filtered

@st.cache_data
def prepare_models(df_filtered):
    user_product_matrix = df_filtered.pivot_table(
        index='userId',
        columns='productId',
        values='rating'
    ).fillna(0)
    
    matrix_normalized = normalize(user_product_matrix)
    
    svd = TruncatedSVD(n_components=10, random_state=42)
    matrix_reduced = svd.fit_transform(matrix_normalized)
    
    linked = linkage(matrix_reduced, method='ward')
    hierarchical_labels = fcluster(linked, t=2, criterion='maxclust')
    
    hierarchical_clusters = pd.DataFrame({
        'userId': user_product_matrix.index,
        'cluster': hierarchical_labels
    })
    
    user_similarity = cosine_similarity(user_product_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_product_matrix.index,
        columns=user_product_matrix.index
    )
    
    return user_product_matrix, hierarchical_clusters, user_similarity_df

def get_recommendations(userId, user_product_matrix, hierarchical_clusters, user_similarity_df, n=10):
    if userId not in user_product_matrix.index:
        return None, None
    
    user_cluster = hierarchical_clusters[
        hierarchical_clusters['userId'] == userId
    ]['cluster'].values[0]
    
    cluster_users = hierarchical_clusters[
        hierarchical_clusters['cluster'] == user_cluster
    ]['userId'].values
    cluster_users = [u for u in cluster_users if u in user_product_matrix.index]
    
    similar_users = user_similarity_df[userId][cluster_users].sort_values(ascending=False)[1:11]
    similar_users_products = user_product_matrix.loc[similar_users.index]
    
    already_rated = user_product_matrix.loc[userId][user_product_matrix.loc[userId] > 0].index
    
    mean_ratings = similar_users_products.mean(axis=0)
    recommendations = mean_ratings.drop(already_rated, errors='ignore')
    recommendations = recommendations[recommendations > 0]
    recommendations = recommendations.sort_values(ascending=False).head(n)
    
    return recommendations, user_cluster

# Streamlit App
st.title("🛍️ Product Recommendation System")
st.subheader("P674 - Product Recommendation System")

st.write("---")

with st.spinner("Loading data and building models... Please wait!"):
    df_filtered = load_data()
    user_product_matrix, hierarchical_clusters, user_similarity_df = prepare_models(df_filtered)

st.success("Models loaded successfully!")

st.write("---")

st.sidebar.title("ℹ️ About")
st.sidebar.write("This recommendation system uses:")
st.sidebar.write("✅ Hierarchical Clustering")
st.sidebar.write("✅ Collaborative Filtering")
st.sidebar.write("✅ Amazon Electronics Dataset")
st.sidebar.write(f"📊 Total Users: {df_filtered['userId'].nunique()}")
st.sidebar.write(f"📦 Total Products: {df_filtered['productId'].nunique()}")
st.sidebar.write(f"⭐ Total Ratings: {len(df_filtered)}")

st.header("Get Product Recommendations")

userId = st.text_input("Enter your User ID:", placeholder="e.g. A100UD67AHFODS")

n_recommendations = st.slider("Number of recommendations:", min_value=5, max_value=20, value=10)

if st.button("Get Recommendations! 🚀"):
    if userId == "":
        st.warning("Please enter a User ID!")
    else:
        with st.spinner("Finding recommendations..."):
            recommendations, user_cluster = get_recommendations(
                userId,
                user_product_matrix,
                hierarchical_clusters,
                user_similarity_df,
                n_recommendations
            )
        
        if recommendations is None:
            st.error(f"User ID '{userId}' not found in our database!")
            st.info("Try one of these user IDs: " + ", ".join(list(user_product_matrix.index[:3])))
        else:
            st.success(f"Found recommendations for User: {userId}")
            st.info(f"You belong to Cluster {user_cluster}")
            
            st.subheader(f"Top {n_recommendations} Recommended Products:")
            
            rec_df = pd.DataFrame({
                'Rank': range(1, len(recommendations) + 1),
                'Product ID': recommendations.index,
                'Recommendation Score': recommendations.values.round(4)
            })
            
            st.dataframe(rec_df, use_container_width=True)
            
            st.subheader("Recommendation Scores Chart:")
            st.bar_chart(recommendations)

st.write("---")
st.caption("P674 Product Recommendation System | Built with Streamlit")