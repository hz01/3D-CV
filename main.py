from colmap_pipeline import ColmapPipeline


def main():
    """Main function to run the COLMAP pipeline."""
    
    # Initialize the pipeline (CUDA will be auto-detected if available)
    pipeline = ColmapPipeline(
        project_root="colmap_cpu_project",
        num_threads=6,
    )
    
    # Run pipeline steps in order
    print("=" * 60)
    print("STEP 1: Download and extract dataset")
    print("=" * 60)
    pipeline.download_and_extract_dataset()
    
    print("\n" + "=" * 60)
    print("STEP 2: Extract features")
    print("=" * 60)
    pipeline.extract_features()
    
    print("\n" + "=" * 60)
    print("STEP 3: Match features")
    print("=" * 60)
    pipeline.match_features()
    
    print("\n" + "=" * 60)
    print("STEP 4: Run Structure-from-Motion (SfM)")
    print("=" * 60)
    pipeline.run_sfm()
    
    print("\n" + "=" * 60)
    print("STEP 5: Visualize sparse reconstruction")
    print("=" * 60)
    # Uncomment the line below if you want to visualize
    # pipeline.visualize_sparse()

    print("\n" + "=" * 60)
    print("STEP 6: Export sparse point cloud to PLY")
    print("=" * 60)
    pipeline.export_sparse_ply()
    
    print("\n" + "=" * 60)
    print("STEP 7: Run dense reconstruction (MVS)")
    print("=" * 60)
    pipeline.run_dense()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
