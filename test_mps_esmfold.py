#!/usr/bin/env python3
"""
Test script to verify MPS support for ESMFold
"""

import torch
import esm
import time
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_device_availability():
    """Test which devices are available"""
    logger.info("=" * 60)
    logger.info("Testing device availability")
    logger.info("=" * 60)
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Determine best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"✅ CUDA GPU is available")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"✅ Apple Metal Performance Shaders (MPS) is available")
    else:
        device = torch.device("cpu")
        logger.info(f"ℹ️ Using CPU (no GPU acceleration available)")
    
    return device

def test_esmfold_on_device(device):
    """Test ESMFold model on specified device"""
    logger.info("=" * 60)
    logger.info(f"Testing ESMFold on {device}")
    logger.info("=" * 60)
    
    try:
        # Load model
        logger.info("Loading ESMFold v1 model...")
        model = esm.pretrained.esmfold_v1()
        model = model.eval()
        
        # Move model to device
        if device.type == "mps":
            logger.info("Converting model to fp32 for MPS compatibility...")
            model.esm.float()  # MPS works better with fp32
            model.to(device)
            logger.info(f"✅ Model successfully moved to MPS")
        elif device.type == "cuda":
            model.cuda()
            logger.info(f"✅ Model successfully moved to CUDA")
        else:
            model.esm.float()  # CPU also needs fp32
            model.cpu()
            logger.info(f"✅ Model on CPU")
        
        # Test with a small protein sequence
        test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
        logger.info(f"Testing inference with sequence of length {len(test_sequence)}")
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            output = model.infer_pdb(test_sequence)
        inference_time = time.time() - start_time
        
        logger.info(f"✅ Inference successful on {device}")
        logger.info(f"Inference time: {inference_time:.2f} seconds")
        
        # Basic validation of output
        if output and len(output) > 100:  # PDB files are typically much longer
            logger.info(f"✅ Generated PDB output of length {len(output)}")
        else:
            logger.warning(f"⚠️ Generated PDB seems too short: {len(output)} characters")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing ESMFold on {device}: {e}")
        return False

def test_batch_inference(device):
    """Test batch inference on the device"""
    logger.info("=" * 60)
    logger.info(f"Testing batch inference on {device}")
    logger.info("=" * 60)
    
    try:
        model = esm.pretrained.esmfold_v1()
        model = model.eval()
        
        # Setup device
        if device.type == "mps":
            model.esm.float()
            model.to(device)
        elif device.type == "cuda":
            model.cuda()
        else:
            model.esm.float()
            model.cpu()
        
        # Test with multiple sequences
        test_sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"
        ]
        
        logger.info(f"Testing batch inference with {len(test_sequences)} sequences")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.infer(test_sequences)
        batch_time = time.time() - start_time
        
        logger.info(f"✅ Batch inference successful on {device}")
        logger.info(f"Batch inference time: {batch_time:.2f} seconds")
        logger.info(f"Average time per sequence: {batch_time/len(test_sequences):.2f} seconds")
        
        # Check outputs
        if "ptm" in outputs:
            logger.info(f"pTM scores: {outputs['ptm'].cpu().numpy()}")
        if "mean_plddt" in outputs:
            logger.info(f"Mean pLDDT scores: {outputs['mean_plddt'].cpu().numpy()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in batch inference on {device}: {e}")
        return False

def compare_devices():
    """Compare inference speed across available devices"""
    logger.info("=" * 60)
    logger.info("Comparing inference speed across devices")
    logger.info("=" * 60)
    
    test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    results = {}
    
    # Test on CPU
    try:
        logger.info("Testing on CPU...")
        model = esm.pretrained.esmfold_v1()
        model = model.eval()
        model.esm.float()
        model.cpu()
        
        start_time = time.time()
        with torch.no_grad():
            _ = model.infer_pdb(test_sequence)
        cpu_time = time.time() - start_time
        results['cpu'] = cpu_time
        logger.info(f"CPU inference time: {cpu_time:.2f} seconds")
        del model  # Free memory
    except Exception as e:
        logger.error(f"CPU test failed: {e}")
    
    # Test on MPS if available
    if torch.backends.mps.is_available():
        try:
            logger.info("Testing on MPS...")
            model = esm.pretrained.esmfold_v1()
            model = model.eval()
            model.esm.float()
            model.to('mps')
            
            # Warm-up run
            with torch.no_grad():
                _ = model.infer_pdb(test_sequence)
            
            start_time = time.time()
            with torch.no_grad():
                _ = model.infer_pdb(test_sequence)
            mps_time = time.time() - start_time
            results['mps'] = mps_time
            logger.info(f"MPS inference time: {mps_time:.2f} seconds")
            
            if 'cpu' in results:
                speedup = results['cpu'] / mps_time
                logger.info(f"MPS speedup over CPU: {speedup:.2f}x")
            del model  # Free memory
        except Exception as e:
            logger.error(f"MPS test failed: {e}")
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        try:
            logger.info("Testing on CUDA...")
            model = esm.pretrained.esmfold_v1()
            model = model.eval()
            model.cuda()
            
            # Warm-up run
            with torch.no_grad():
                _ = model.infer_pdb(test_sequence)
            torch.cuda.synchronize()
            
            start_time = time.time()
            with torch.no_grad():
                _ = model.infer_pdb(test_sequence)
            torch.cuda.synchronize()
            cuda_time = time.time() - start_time
            results['cuda'] = cuda_time
            logger.info(f"CUDA inference time: {cuda_time:.2f} seconds")
            
            if 'cpu' in results:
                speedup = results['cpu'] / cuda_time
                logger.info(f"CUDA speedup over CPU: {speedup:.2f}x")
            del model  # Free memory
        except Exception as e:
            logger.error(f"CUDA test failed: {e}")
    
    return results

def main():
    """Main test function"""
    logger.info("Starting ESMFold MPS Support Tests")
    logger.info("=" * 60)
    
    # Test 1: Check device availability
    device = test_device_availability()
    
    # Test 2: Test ESMFold on detected device
    if test_esmfold_on_device(device):
        logger.info("✅ Basic ESMFold test passed")
    else:
        logger.error("❌ Basic ESMFold test failed")
        sys.exit(1)
    
    # Test 3: Test batch inference
    if test_batch_inference(device):
        logger.info("✅ Batch inference test passed")
    else:
        logger.error("❌ Batch inference test failed")
    
    # Test 4: Compare devices (if multiple available)
    logger.info("\n")
    results = compare_devices()
    
    logger.info("=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)
    
    # Summary
    if results:
        logger.info("Performance Summary:")
        for device_name, time_taken in results.items():
            logger.info(f"  {device_name.upper()}: {time_taken:.2f} seconds")

if __name__ == "__main__":
    main()