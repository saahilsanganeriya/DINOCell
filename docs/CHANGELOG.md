# DINOCell Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-01-XX - Initial Release

### Added

#### Core Framework
- **DINOCell Model Architecture**
  - DINOv3 ViT backbone with multi-scale feature extraction
  - U-Net style decoder for distance map prediction
  - Support for Small/Base/Large/7B model variants
  - Flexible fine-tuning (freeze or train backbone)

- **Inference Pipeline**
  - Sliding window inference with 256×256 patches
  - 32-pixel overlap with cosine blending
  - CLAHE preprocessing for contrast enhancement
  - Watershed post-processing for cell mask extraction
  - Compatible interface with SAMCell

- **Training Framework**
  - MSE loss for distance map regression
  - AdamW optimizer with learning rate warmup
  - Early stopping with configurable patience
  - Multi-dataset support (concatenate multiple datasets)
  - Comprehensive logging and checkpointing

#### Dataset Support
- **Dataset Processing Scripts**
  - LIVECell (COCO format) processor
  - Cellpose (numbered pairs) processor
  - Custom dataset processor
  - Automatic distance map generation
  - Support for variable image sizes

- **Data Augmentation**
  - Random horizontal flip
  - Random rotation (-180° to 180°)
  - Random scale (0.8-1.2×)
  - Random brightness (0.95-1.05×)
  - Random image inversion

#### Evaluation
- **Evaluation Scripts**
  - Cell Tracking Challenge metrics (SEG, DET, OP_CSB)
  - Threshold grid search for parameter tuning
  - Batch evaluation on multiple datasets
  - Results export to CSV

- **Compatible Datasets**
  - LIVECell (5000+ images, 8 cell types)
  - Cellpose Cytoplasm (~600 images)
  - PBL-HEK (5 images, zero-shot evaluation)
  - PBL-N2a (5 images, zero-shot evaluation)

#### Documentation
- README.md with quick start guide
- GETTING_STARTED.md for new users
- TRAINING_GUIDE.md with detailed training instructions
- DATASETS.md with dataset information
- ARCHITECTURE.md with technical details
- PROJECT_SUMMARY.md with complete overview
- Tutorial.ipynb for interactive learning

#### Examples
- simple_inference.py: Basic usage example
- compare_with_samcell.py: Side-by-side comparison
- CLI tool for command-line usage

#### Configuration
- YAML config support for reproducible training
- Reasonable defaults based on SAMCell
- Easy customization of all hyperparameters

### Design Philosophy

- **Compatibility**: Works with SAMCell datasets and evaluation
- **Modularity**: Clean separation of components
- **Extensibility**: Easy to modify and extend
- **Documentation**: Comprehensive docs for all levels
- **Best Practices**: Type hints, logging, error handling

### Dependencies

- PyTorch ≥2.0.0
- DINOv3 (from Meta AI)
- OpenCV for image processing
- scikit-image for watershed
- Standard scientific Python stack

### Known Limitations

- Requires DINOv3 repository in parent directory
- Large model variants (7B) require significant compute
- Cell Tracking Challenge binaries needed for official metrics
- Currently tested on Linux/macOS (Windows support TBD)

## Future Releases

### Planned for v0.2.0

- [ ] Pre-trained model weights (DINOCell-Generalist)
- [ ] Benchmark results on all datasets
- [ ] Performance comparison with SAMCell
- [ ] GUI application (similar to SAMCell-GUI)
- [ ] Napari plugin
- [ ] Docker container for easy deployment
- [ ] Colab notebook for cloud usage

### Planned for v0.3.0

- [ ] 3D cell segmentation support
- [ ] Multi-channel image support (fluorescence)
- [ ] Cell tracking in time-lapse videos
- [ ] Interactive segmentation with prompts
- [ ] Model ensembling support
- [ ] TensorRT/ONNX export for optimization

### Research Features (Future)

- [ ] Multi-task learning (segmentation + classification)
- [ ] Uncertainty estimation
- [ ] Few-shot learning support
- [ ] Domain adaptation techniques
- [ ] Automated hyperparameter tuning

## Notes

This is the initial release of DINOCell. The framework is fully functional and ready for
research and development. Benchmark results and pre-trained weights will be added soon.

For the latest updates, check the GitHub repository.



