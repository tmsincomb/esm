#!/usr/bin/env python3
"""
Simple test script to verify MPS support for ESM models (not ESMFold)
"""

import torch
import esm
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_esm2_on_mps():
    """Test ESM-2 model on MPS"""
    logger.info("=" * 60)
    logger.info("Testing ESM-2 on MPS (without ESMFold)")
    logger.info("=" * 60)
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"MPS built: {torch.backends.mps.is_built()}")
    
    if not torch.backends.mps.is_available():
        logger.error("MPS is not available on this system")
        return False
    
    try:
        # Load a smaller ESM-2 model for testing
        logger.info("Loading ESM-2 model (8M parameters)...")
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        
        # Move model to MPS
        logger.info("Moving model to MPS...")
        device = torch.device("mps")
        model = model.to(device)
        logger.info("✅ Model successfully moved to MPS")
        
        # Prepare test data
        data = [
            ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
            ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ]
        
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        
        # Test inference on MPS
        logger.info("Running inference on MPS...")
        start_time = time.time()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])
        mps_time = time.time() - start_time
        
        logger.info(f"✅ MPS inference successful!")
        logger.info(f"Inference time on MPS: {mps_time:.3f} seconds")
        
        # Test on CPU for comparison
        logger.info("Running same inference on CPU for comparison...")
        model_cpu = model.cpu()
        batch_tokens_cpu = batch_tokens.cpu()
        
        start_time = time.time()
        with torch.no_grad():
            results_cpu = model_cpu(batch_tokens_cpu, repr_layers=[6])
        cpu_time = time.time() - start_time
        
        logger.info(f"Inference time on CPU: {cpu_time:.3f} seconds")
        
        # Calculate speedup
        speedup = cpu_time / mps_time
        logger.info(f"🚀 MPS speedup over CPU: {speedup:.2f}x")
        
        # Verify results are similar
        mps_output = results["representations"][6].cpu()
        cpu_output = results_cpu["representations"][6]
        
        diff = torch.abs(mps_output - cpu_output).max().item()
        logger.info(f"Maximum difference between MPS and CPU outputs: {diff:.6f}")
        
        if diff < 0.01:  # Tolerance for floating point differences
            logger.info("✅ MPS and CPU outputs are consistent")
        else:
            logger.warning(f"⚠️ MPS and CPU outputs differ by {diff}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing ESM-2 on MPS: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_model_sizes():
    """Test different ESM-2 model sizes on MPS"""
    logger.info("=" * 60)
    logger.info("Testing different ESM-2 model sizes on MPS")
    logger.info("=" * 60)
    
    if not torch.backends.mps.is_available():
        logger.error("MPS is not available")
        return
    
    models_to_test = [
        ("esm2_t6_8M_UR50D", "ESM-2 8M"),
        ("esm2_t12_35M_UR50D", "ESM-2 35M"),
    ]
    
    test_sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    device = torch.device("mps")
    
    for model_name, description in models_to_test:
        try:
            logger.info(f"\nTesting {description}...")
            
            # Load model
            load_fn = getattr(esm.pretrained, model_name)
            model, alphabet = load_fn()
            batch_converter = alphabet.get_batch_converter()
            model.eval()
            
            # Move to MPS
            model = model.to(device)
            
            # Prepare data
            data = [("test", test_sequence)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                _ = model(batch_tokens)
            inference_time = time.time() - start_time
            
            logger.info(f"✅ {description} on MPS: {inference_time:.3f} seconds")
            
            # Clean up
            del model
            torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
            
        except Exception as e:
            logger.error(f"❌ Failed to test {description}: {e}")

def main():
    """Main test function"""
    logger.info("Starting ESM-2 MPS Support Tests (without ESMFold)")
    logger.info("=" * 60)
    
    # Test basic ESM-2 on MPS
    if test_esm2_on_mps():
        logger.info("✅ Basic ESM-2 MPS test passed")
    else:
        logger.error("❌ Basic ESM-2 MPS test failed")
    
    # Test different model sizes
    test_different_model_sizes()
    
    logger.info("=" * 60)
    logger.info("All tests completed!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()