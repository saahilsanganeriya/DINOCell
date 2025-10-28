# Complete Requirements - All User Requests Consolidated

This document consolidates every request made during our conversation.

## 📋 Request 1: Create DINOCell Framework (Initial)

**Task**: Build complete DINOCell framework using DINOv3 instead of SAM

**Requirements**:
1. Read and understand entirety of SAMCell framework
2. Read SAMCell manuscript (main.tex and s1_appendix.tex) to understand publication
3. Create new framework called DINOCell that:
   - Uses DINOv3 as base instead of SAM
   - Uses same watershed approach for post-processing
   - Outputs distance maps (same as SAMCell)
   - Uses U-Net type architecture or whatever is best for segmentation
4. Implement complete project:
   - Training scripts
   - Evaluation scripts  
   - Dataset processing utilities
   - Entire framework structure
5. Use LIVECell and Cellpose datasets for fine-tuning

---

## 📋 Request 2: SSL Pretraining Implementation

**Task**: Implement DINOv3 self-supervised pretraining on JUMP Cell Painting dataset

**Requirements**:
1. Use JUMP Cell Painting dataset (3 million images with 5 fluorescent + 3 brightfield channels)
2. Find big dataset of cell images for pretraining (they don't need to be segmented)
3. Implement pretraining approach where:
   - Each field has 5 channels and 3 brightfield
   - Get same 'output' for all channels (they're same cells, just different stains)
   - Later can finetune distance map head separately for each channel type
4. Decide on appropriate model for dataset (with only A100 for 24-48hrs)
5. Decide: Continue from pretrained checkpoint or train from scratch?
6. Use patch size 8 (not 16) for higher resolution/precision
7. Create config file
8. Create training scripts
9. Make complete pretraining guide

---

## 📋 Request 3: S3 Streaming & Repository Organization

**Three sub-requests**:

### 3A: AWS S3 Streaming
**Requirements**:
1. Stream images from AWS S3 (don't save all images locally)
2. JUMP dataset is in AWS
3. Implement streaming solution

### 3B: Repository Organization
**Requirements**:
1. DINOv3 folder and DINOCell folder have both been edited but are part of different repos
2. Combine them since we edited the DINOv3 repo
3. Create unified repository structure

### 3C: Wandb Logging
**Requirements**:
1. Log training metrics periodically (loss, learning rate)
2. Log if model is getting better
3. Log appropriate visualizations:
   - Attention maps
   - Any other relevant visualizations
4. Integrate wandb into training pipeline

---

## 📋 Request 4: Interview Preparation Documentation

**Task**: Create comprehensive interview preparation documentation

**Requirements**:

### For SAMCell (Published Work):
1. **Technical Narrative Document**:
   - Explain all technical decisions in narrative format
   - Refer to paper for details (don't make anything up, don't hallucinate)
   - Explain key technical decisions

2. **Q&A Document**:
   - Create questions a computer vision engineer interviewer might ask
   - Provide answers for each question
   - Focus on SAMCell project

### For DINOCell (Current Work):
1. **Consolidate Documents**:
   - Combine the huge set of documents into <5 docs
   - Add diagrams (mermaid or otherwise)
   - Include:
     - Architecture explanation
     - Pretraining procedure explanation
     - All technical details

2. **Create Diagrams**:
   - Make diagrams about the pretraining process
   - Make diagrams about the U-Net architecture
   - Use mermaid diagrams etc.

3. **Q&A Document**:
   - Create huge set of possible technical questions a CV engineer might ask
   - Answer all questions
   - Focus on DINOCell and SAMCell projects
   - Prepare for showcasing SAMCell as published work
   - Prepare for showcasing DINOCell as current work

### General Interview Prep Requirements:
1. Reading all these docs should prepare for interview
2. Focus on the projects (not the company - Duranta)
3. Questions should be what can be asked in a technical interview
4. Don't hallucinate - base everything on actual implementation and papers

---

## 🎯 Context Provided

### Job Description (Duranta):
- Computer Vision Engineer role
- Seed-stage startup
- Focus on: Semantic/instance segmentation, few-shot learning, 3D reasoning, generative modeling
- Working with: Aerial and oblique imagery
- Models: Mask2Former, SAM, ViT, Swin
- Requirements: 2+ years ML/DL experience, trained segmentation models at scale
- Extra credit: Self-supervised and weakly-supervised learning, AWS experience

### Datasets Referenced:
- **SAMCell**: LIVECell (5K images), Cellpose Cytoplasm (600 images), PBL-HEK (5 images), PBL-N2a (5 images)
- **DINOCell**: Same as above + JUMP Cell Painting (3M images, 5 fluorescent + 3 brightfield channels per field)

### Papers Referenced:
- SAMCell manuscript: main.tex and s1_appendix.tex
- JUMP Cell Painting: 2024_Chandrasekaran_NatureMethods_CPJUMP1

---

## ✅ Summary of Complete Requirements

**Create**:
1. ✅ Complete DINOCell framework (architecture, training, evaluation)
2. ✅ SSL pretraining on JUMP with multi-view consistency learning
3. ✅ S3 streaming implementation (no local 500GB download)
4. ✅ Unified repository structure (combine DINOv3 + DINOCell)
5. ✅ Wandb integration for training monitoring
6. ✅ SAMCell technical narrative document
7. ✅ SAMCell interview Q&A
8. ✅ DINOCell consolidated documentation (<5 docs with diagrams)
9. ✅ DINOCell architecture diagrams (mermaid)
10. ✅ DINOCell pretraining diagrams (mermaid)
11. ✅ DINOCell interview Q&A (comprehensive technical questions)

**Documentation Must**:
- Be factually accurate (no hallucinations)
- Reference papers for details
- Include mermaid diagrams
- Prepare user for technical interview
- Focus on showcasing projects, not the company
- Cover both SAMCell (published) and DINOCell (current work)

