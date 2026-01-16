import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import umap
import warnings
warnings.filterwarnings('ignore')


def load_latent_vectors(output_dir="outputs"):
    train_latents = np.load(f"{output_dir}/train_latents.npy")
    val_latents = np.load(f"{output_dir}/val_latents.npy")
    test_latents = np.load(f"{output_dir}/test_latents.npy")
    all_latents = np.vstack([train_latents, val_latents, test_latents])
    
    print(f"Loaded {all_latents.shape[0]} latent vectors ({all_latents.shape[1]} dims)")
    
    splits = {
        'train': (0, len(train_latents)),
        'val': (len(train_latents), len(train_latents) + len(val_latents)),
        'test': (len(train_latents) + len(val_latents), len(all_latents))
    }
    
    return all_latents, splits


def kmeans_baseline(latents, k_range=[2, 3, 4, 5], save_dir="outputs/clustering"):
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    results = {}
    
    print("\nK-MEANS:")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=500)
        labels = kmeans.fit_predict(latents)
        silhouette = silhouette_score(latents, labels)
        
        results[k] = {
            'model': kmeans,
            'labels': labels,
            'inertia': kmeans.inertia_,
            'silhouette': silhouette
        }
        
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  K={k}: silhouette={silhouette:.4f}, inertia={kmeans.inertia_:.4f}, sizes={dict(zip(unique, counts))}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    inertias = [results[k]['inertia'] for k in k_range]
    silhouettes = [results[k]['silhouette'] for k in k_range]
    
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('K')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Curve')
    ax1.grid(alpha=0.3)
    
    ax2.plot(k_range, silhouettes, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('K')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Scores')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/kmeans_elbow_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    with open(f"{save_dir}/kmeans_summary.json", 'w') as f:
        json.dump({k: {'inertia': float(results[k]['inertia']), 
                       'silhouette': float(results[k]['silhouette'])} 
                   for k in k_range}, f, indent=2)
    
    return results


def gmm_clustering(latents, k_range=[2, 3, 4, 5], save_dir="outputs/clustering"):
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    results = {}
    
    print("\nGMM:")
    for k in k_range:
        best_gmm = None
        best_bic = float('inf')
        
        for cov_type in ['full', 'tied', 'diag', 'spherical']:
            try:
                gmm = GaussianMixture(n_components=k, covariance_type=cov_type, random_state=42,
                                     n_init=10, max_iter=500, init_params='k-means++', reg_covar=1e-6)
                gmm.fit(latents)
                bic_score = gmm.bic(latents)
                if bic_score < best_bic:
                    best_bic = bic_score
                    best_gmm = gmm
            except:
                continue
        
        if best_gmm is None:
            print(f"  K={k}: FAILED")
            continue
        
        labels = best_gmm.predict(latents)
        proba = best_gmm.predict_proba(latents)
        n_unique = len(np.unique(labels))
        silhouette = silhouette_score(latents, labels) if n_unique > 1 else -1.0
        
        results[k] = {
            'model': best_gmm,
            'labels': labels,
            'proba': proba,
            'bic': best_gmm.bic(latents),
            'aic': best_gmm.aic(latents),
            'silhouette': silhouette
        }
        
        unique, counts = np.unique(labels, return_counts=True)
        status = " [COLLAPSED]" if n_unique == 1 else ""
        print(f"  K={k}: BIC={results[k]['bic']:.2f}, AIC={results[k]['aic']:.2f}, "
              f"silhouette={silhouette:.4f}, sizes={dict(zip(unique, counts))}{status}")
    
    if not results:
        print("GMM failed for all K")
        return {}, None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    bics = [results[k]['bic'] for k in k_range if k in results]
    aics = [results[k]['aic'] for k in k_range if k in results]
    silhouettes = [results[k]['silhouette'] for k in k_range if k in results]
    valid_k = [k for k in k_range if k in results]
    
    ax1.plot(valid_k, bics, 'ro-', linewidth=2, markersize=8, label='BIC')
    ax1.plot(valid_k, aics, 'bo-', linewidth=2, markersize=8, label='AIC')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Information Criterion')
    ax1.set_title('GMM Model Selection (lower=better)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(valid_k, silhouettes, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('K')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('GMM Silhouette')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/gmm_model_selection.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    best_k = valid_k[np.argmin(bics)]
    print(f"\nBest GMM: K={best_k} (BIC={results[best_k]['bic']:.2f})")
    
    with open(f"{save_dir}/best_gmm_model.pkl", 'wb') as f:
        pickle.dump(results[best_k]['model'], f)
    
    for k in results:
        np.save(f"{save_dir}/gmm_labels_k{k}.npy", results[k]['labels'])
        np.save(f"{save_dir}/gmm_proba_k{k}.npy", results[k]['proba'])
    
    with open(f"{save_dir}/gmm_summary.json", 'w') as f:
        json.dump({k: {'bic': float(results[k]['bic']), 'aic': float(results[k]['aic']),
                       'silhouette': float(results[k]['silhouette'])} 
                   for k in results}, f, indent=2)
    
    return results, best_k


def visualize_clusters(latents, labels, save_dir="outputs/clustering"):
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    print("\nGenerating visualizations...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    latents_tsne = tsne.fit_transform(latents)
    
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    latents_umap = reducer.fit_transform(latents)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    scatter1 = axes[0, 0].scatter(latents_tsne[:, 0], latents_tsne[:, 1], 
                                  c=labels, cmap='viridis', alpha=0.6, s=10)
    axes[0, 0].set_title('t-SNE Projection')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    scatter2 = axes[0, 1].scatter(latents_umap[:, 0], latents_umap[:, 1],
                                  c=labels, cmap='viridis', alpha=0.6, s=10)
    axes[0, 1].set_title('UMAP Projection')
    axes[0, 1].set_xlabel('UMAP 1')
    axes[0, 1].set_ylabel('UMAP 2')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    unique, counts = np.unique(labels, return_counts=True)
    axes[1, 0].bar(unique, counts, color='steelblue', alpha=0.7)
    axes[1, 0].set_title('Cluster Sizes')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(alpha=0.3, axis='y')
    for cluster, count in zip(unique, counts):
        axes[1, 0].text(cluster, count, str(count), ha='center', va='bottom')
    
    cluster_means = np.array([latents[labels == i].mean(axis=0) for i in unique])
    im = axes[1, 1].imshow(cluster_means, cmap='coolwarm', aspect='auto')
    axes[1, 1].set_title('Mean Latent Features per Cluster')
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('Cluster')
    axes[1, 1].set_yticks(range(len(unique)))
    axes[1, 1].set_yticklabels(unique)
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cluster_visualizations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    np.save(f"{save_dir}/tsne_coords.npy", latents_tsne)
    np.save(f"{save_dir}/umap_coords.npy", latents_umap)


def cluster_characterization(latents, labels, splits, save_dir="outputs/clustering"):
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    n_clusters = len(np.unique(labels))
    cluster_stats = {}
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_latents = latents[mask]
        if len(cluster_latents) == 0:
            continue
        
        cluster_stats[cluster_id] = {
            'size': int(mask.sum()),
            'mean': cluster_latents.mean(axis=0).tolist(),
            'std': cluster_latents.std(axis=0).tolist()
        }
    
    temporal_dist = {}
    for split_name, (start, end) in splits.items():
        split_labels = labels[start:end]
        unique, counts = np.unique(split_labels, return_counts=True)
        temporal_dist[split_name] = {int(k): int(v) for k, v in zip(unique, counts)}
    
    print("\nTemporal distribution:")
    for cluster_id in sorted(cluster_stats.keys()):
        print(f"  Cluster {cluster_id}:", end="")
        for split in ['train', 'val', 'test']:
            count = temporal_dist[split].get(cluster_id, 0)
            total = sum(temporal_dist[split].values())
            pct = 100 * count / total if total > 0 else 0
            print(f" {split}={count}({pct:.0f}%)", end="")
        print()
    
    with open(f"{save_dir}/cluster_characterization.json", 'w') as f:
        json.dump({'cluster_stats': cluster_stats, 'temporal_distribution': temporal_dist}, f, indent=2)


def main():
    save_dir = "outputs/clustering"
    all_latents, splits = load_latent_vectors()
    
    kmeans_results = kmeans_baseline(all_latents, k_range=[2, 3, 4, 5], save_dir=save_dir)
    
    gmm_results, best_k = gmm_clustering(all_latents, k_range=[2, 3, 4, 5], save_dir=save_dir)
    
    if best_k and len(np.unique(gmm_results[best_k]['labels'])) > 1:
        labels = gmm_results[best_k]['labels']
        method = "GMM"
    else:
        best_k = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]['silhouette'])
        labels = kmeans_results[best_k]['labels']
        method = "K-Means"
    
    print(f"\nUsing {method} K={best_k} for analysis")
    
    visualize_clusters(all_latents, labels, save_dir=save_dir)
    
    cluster_characterization(all_latents, labels, splits, save_dir=save_dir)

if __name__ == "__main__":
    main()
