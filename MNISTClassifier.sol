// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MNISTClassifierMini {
    // Model architecture parameters 
    uint16 constant INPUT_SIZE = 400;    // 20x20 images
    uint16 constant HIDDEN_SIZE1 = 32;   // First hidden layer
    uint8 constant HIDDEN_SIZE2 = 16;    // Second hidden layer
    uint8 constant OUTPUT_SIZE = 10;     // 10 digits (0-9)
    
    // Weights
    mapping(uint32 => int8) public inputToHidden1Weights;
    int8[32] private hidden1Biases;
    mapping(uint32 => int8) public hidden1ToHidden2Weights;
    int8[16] private hidden2Biases;
    mapping(uint32 => int8) public hidden2ToOutputWeights;
    int8[10] private outputBiases;
    
    // Scaling factor for quantization
    int32 private scaleFactor;
    
    // Deployment state tracking (TODO: remove)
    bool private biasesInitialized = false;
    mapping(uint16 => bool) private w1ChunksUploaded;
    uint16 private w1ChunksCount;
    mapping(uint16 => bool) private w2ChunksUploaded;
    uint16 private w2ChunksCount;
    mapping(uint16 => bool) private w3ChunksUploaded;
    uint16 private w3ChunksCount;
    
    event PredictionResult(uint8 digit);
    
    constructor(int32 _scaleFactor, uint16 _w1ChunksCount, uint16 _w2ChunksCount, uint16 _w3ChunksCount) {
        scaleFactor = _scaleFactor;
        w1ChunksCount = _w1ChunksCount;
        w2ChunksCount = _w2ChunksCount;
        w3ChunksCount = _w3ChunksCount;
    }
    
    function initializeBiases(
        int8[32] calldata _hidden1Biases,
        int8[16] calldata _hidden2Biases,
        int8[10] calldata _outputBiases
    ) external {
        require(!biasesInitialized, "Biases already initialized");
        
        for(uint i = 0; i < 32; i++) {
            hidden1Biases[i] = _hidden1Biases[i];
        }
        
        for(uint i = 0; i < 16; i++) {
            hidden2Biases[i] = _hidden2Biases[i];
        }
        
        for(uint i = 0; i < 10; i++) {
            outputBiases[i] = _outputBiases[i];
        }
        
        biasesInitialized = true;
    }

    function uploadInputHidden1Weights(
        uint16 chunkIndex, 
        int8[] calldata weights,
        uint32 startPos
    ) external {
        require(biasesInitialized, "Initialize biases first");
        require(chunkIndex < w1ChunksCount, "Invalid chunk index");
        require(!w1ChunksUploaded[chunkIndex], "Chunk already uploaded");
        require(startPos + weights.length <= uint32(INPUT_SIZE) * uint32(HIDDEN_SIZE1), "Weights overflow");
        require(weights.length <= 500, "Chunk too large"); // Strict limit on chunk size
        
        for (uint i = 0; i < weights.length; i++) {
            inputToHidden1Weights[startPos + uint32(i)] = weights[i];
        }
        
        w1ChunksUploaded[chunkIndex] = true;
    }

    function uploadHidden1Hidden2Weights(
        uint16 chunkIndex,
        int8[] calldata weights,
        uint32 startPos
    ) external {
        require(biasesInitialized, "Initialize biases first");
        require(chunkIndex < w2ChunksCount, "Invalid chunk index");
        require(!w2ChunksUploaded[chunkIndex], "Chunk already uploaded");
        require(startPos + weights.length <= uint32(HIDDEN_SIZE1) * uint32(HIDDEN_SIZE2), "Weights overflow");
        
        for (uint i = 0; i < weights.length; i++) {
            hidden1ToHidden2Weights[startPos + uint32(i)] = weights[i];
        }
        
        w2ChunksUploaded[chunkIndex] = true;
    }
    
    function uploadHidden2OutputWeights(
        uint16 chunkIndex,
        int8[] calldata weights,
        uint32 startPos
    ) external {
        require(biasesInitialized, "Initialize biases first");
        require(chunkIndex < w3ChunksCount, "Invalid chunk index");
        require(!w3ChunksUploaded[chunkIndex], "Chunk already uploaded");
        require(startPos + weights.length <= uint32(HIDDEN_SIZE2) * uint32(OUTPUT_SIZE), "Weights overflow");
        
        for (uint i = 0; i < weights.length; i++) {
            hidden2ToOutputWeights[startPos + uint32(i)] = weights[i];
        }
        
        w3ChunksUploaded[chunkIndex] = true;
    }
    
    function isModelReady() public view returns (bool) {
        if (!biasesInitialized) return false;
        
        for (uint16 i = 0; i < w1ChunksCount; i++) {
            if (!w1ChunksUploaded[i]) return false;
        }
        
        for (uint16 i = 0; i < w2ChunksCount; i++) {
            if (!w2ChunksUploaded[i]) return false;
        }
        
        for (uint16 i = 0; i < w3ChunksCount; i++) {
            if (!w3ChunksUploaded[i]) return false;
        }
        
        return true;
    }
    
    function relu(int32 x) internal pure returns (int32) {
        return x > 0 ? x : int32(0);
    }
    
    function predict(uint8[400] calldata inputImage) external returns (uint8) {
        require(isModelReady(), "Model not fully uploaded");
        
        // First hidden layer inference
        int32[32] memory hidden1Activations;
        
        for (uint h = 0; h < HIDDEN_SIZE1; h++) {
            int32 sum = int32(hidden1Biases[h]);
            
            for (uint i = 0; i < INPUT_SIZE; i++) {
                uint32 idx = uint32(i) * uint32(HIDDEN_SIZE1) + uint32(h);
                int32 weight = int32(inputToHidden1Weights[idx]);
                
                uint32 pixelValueUint = uint32(inputImage[i]);
                int32 pixelValue = int32(pixelValueUint);
                sum += (pixelValue * weight) / 255;
            }
            
            hidden1Activations[h] = relu(sum);
        }
        
        int32[16] memory hidden2Activations;
        
        for (uint h = 0; h < HIDDEN_SIZE2; h++) {
            int32 sum = int32(hidden2Biases[h]);
            
            for (uint i = 0; i < HIDDEN_SIZE1; i++) {
                uint32 idx = uint32(i) * uint32(HIDDEN_SIZE2) + uint32(h);
                int32 weight = int32(hidden1ToHidden2Weights[idx]);
                sum += (hidden1Activations[i] * weight) / scaleFactor;
            }
            
            hidden2Activations[h] = relu(sum);
        }
        
        // Output layer inference
        int32[10] memory outputActivations;
        
        for (uint o = 0; o < OUTPUT_SIZE; o++) {
            int32 sum = int32(outputBiases[o]);
            
            for (uint h = 0; h < HIDDEN_SIZE2; h++) {
                uint32 idx = uint32(h) * uint32(OUTPUT_SIZE) + uint32(o);
                int32 weight = int32(hidden2ToOutputWeights[idx]);
                sum += (hidden2Activations[h] * weight) / scaleFactor;
            }
            
            outputActivations[o] = sum;
        }
        
        // Find the digit with highest activation (argmax)
        uint8 maxIndex = 0;
        int32 maxValue = outputActivations[0];
        
        for (uint i = 1; i < OUTPUT_SIZE; i++) {
            if (outputActivations[i] > maxValue) {
                maxValue = outputActivations[i];
                maxIndex = uint8(i);
            }
        }
        
        emit PredictionResult(maxIndex);
        return maxIndex;
    }
    
    function getUploadStatus() external view returns (
        bool biasesInit,
        uint16 w1Uploaded,
        uint16 w1Total,
        uint16 w2Uploaded,
        uint16 w2Total,
        uint16 w3Uploaded,
        uint16 w3Total,
        bool ready
    ) {
        uint16 w1Count = 0;
        for (uint16 i = 0; i < w1ChunksCount; i++) {
            if (w1ChunksUploaded[i]) w1Count++;
        }
        
        uint16 w2Count = 0;
        for (uint16 i = 0; i < w2ChunksCount; i++) {
            if (w2ChunksUploaded[i]) w2Count++;
        }
        
        uint16 w3Count = 0;
        for (uint16 i = 0; i < w3ChunksCount; i++) {
            if (w3ChunksUploaded[i]) w3Count++;
        }
        
        return (
            biasesInitialized,
            w1Count,
            w1ChunksCount,
            w2Count,
            w2ChunksCount,
            w3Count,
            w3ChunksCount,
            isModelReady()
        );
    }
}