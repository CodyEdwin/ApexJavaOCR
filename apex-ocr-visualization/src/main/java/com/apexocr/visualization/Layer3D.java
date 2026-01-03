package com.apexocr.visualization;

import com.apexocr.core.monitoring.LayerSnapshot;
import com.apexocr.core.monitoring.NetworkArchitecture;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g3d.Environment;
import com.badlogic.gdx.graphics.g3d.Material;
import com.badlogic.gdx.graphics.g3d.Model;
import com.badlogic.gdx.graphics.g3d.ModelBatch;
import com.badlogic.gdx.graphics.g3d.ModelInstance;
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute;
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder;
import com.badlogic.gdx.graphics.VertexAttributes;
import com.badlogic.gdx.math.Vector3;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a neural network layer in 3D space.
 * Visualizes layer structure, activation intensity, and flow.
 */
public class Layer3D {
    
    private final NetworkArchitecture.LayerInfo layerInfo;
    private final float xPosition;
    
    // 3D components
    private Model mainModel;
    private ModelInstance mainInstance;
    private List<ModelInstance> neuronModels;
    private List<ModelInstance> activationFlowModels;
    
    // Animation
    private float pulsePhase;
    private float activationIntensity;
    
    // Visual properties
    private Color baseColor;
    private Color activeColor;
    private float scale;
    
    public Layer3D(NetworkArchitecture.LayerInfo layerInfo, float xPosition) {
        this.layerInfo = layerInfo;
        this.xPosition = xPosition;
        this.pulsePhase = 0f;
        this.activationIntensity = 0f;
        
        // Set colors based on layer type
        this.baseColor = NetworkRenderer.getLayerColor(layerInfo.type);
        this.activeColor = new Color(
            Math.min(1.0f, baseColor.r + 0.3f),
            Math.min(1.0f, baseColor.g + 0.3f),
            Math.min(1.0f, baseColor.b + 0.3f),
            1.0f
        );
        
        // Calculate scale based on layer dimensions
        this.scale = getLayerSize(layerInfo);
        
        // Build 3D representation
        build3DRepresentation();
    }
    
    private void build3DRepresentation() {
        ModelBuilder modelBuilder = new ModelBuilder();
        neuronModels = new ArrayList<>();
        activationFlowModels = new ArrayList<>();
        
        // Create main layer body
        createLayerBody(modelBuilder);
        
        // Create neuron representations
        createNeurons(modelBuilder);
        
        // Create activation flow visualization
        createActivationFlow(modelBuilder);
    }
    
    private void createLayerBody(ModelBuilder modelBuilder) {
        float layerWidth = 2.0f;
        float layerHeight = getLayerHeight();
        float layerDepth = 4.0f;
        
        Material material = new Material();
        material.set(new ColorAttribute(ColorAttribute.Diffuse, baseColor));
        
        // Create main body with appropriate shape based on layer type
        switch (layerInfo.type) {
            case CONV2D:
                // Create a box for convolutional layers
                mainModel = modelBuilder.createBox(layerWidth, layerHeight, layerDepth, material,
                    VertexAttributes.Usage.Position | VertexAttributes.Usage.Normal);
                break;
                
            case BILSTM:
                // Create a cylinder for LSTM layers (representing recurrent structure)
                mainModel = modelBuilder.createCylinder(layerWidth / 2, layerHeight, layerDepth, 16, material,
                    VertexAttributes.Usage.Position | VertexAttributes.Usage.Normal);
                break;
                
            case DENSE:
            case OUTPUT:
                // Create a sphere for fully connected layers
                mainModel = modelBuilder.createSphere(layerWidth, layerHeight, layerDepth, 16, 16, material,
                    VertexAttributes.Usage.Position | VertexAttributes.Usage.Normal);
                break;
                
            case INPUT:
            default:
                // Create a box for input layers
                mainModel = modelBuilder.createBox(layerWidth, layerHeight, layerDepth, material,
                    VertexAttributes.Usage.Position | VertexAttributes.Usage.Normal);
                break;
        }
        
        mainInstance = new ModelInstance(mainModel);
        mainInstance.transform.setTranslation(new Vector3(xPosition, 0, 0));
    }
    
    private void createNeurons(ModelBuilder modelBuilder) {
        int neuronCount = getNeuronCount();
        float layerHeight = getLayerHeight();
        
        Material material = new Material();
        material.set(new ColorAttribute(ColorAttribute.Diffuse, baseColor));
        
        for (int i = 0; i < neuronCount; i++) {
            float y = -layerHeight / 2 + (i + 1) * (layerHeight / (neuronCount + 1));
            
            // Create small sphere for each neuron
            Model neuronModel = modelBuilder.createSphere(0.3f, 0.3f, 0.3f, 8, 8, material,
                VertexAttributes.Usage.Position | VertexAttributes.Usage.Normal);
            
            ModelInstance neuronInstance = new ModelInstance(neuronModel);
            neuronInstance.transform.setTranslation(new Vector3(xPosition, y, 0));
            neuronModels.add(neuronInstance);
        }
    }
    
    private void createActivationFlow(ModelBuilder modelBuilder) {
        // Create animated particles or lines showing activation flow
        int flowParticles = 10;
        
        Material flowMaterial = new Material();
        flowMaterial.set(new ColorAttribute(ColorAttribute.Emissive, activeColor));
        
        for (int i = 0; i < flowParticles; i++) {
            Model flowModel = modelBuilder.createSphere(0.15f, 0.15f, 0.15f, 6, 6, flowMaterial,
                VertexAttributes.Usage.Position | VertexAttributes.Usage.Normal);
            
            ModelInstance flowInstance = new ModelInstance(flowModel);
            flowInstance.transform.setTranslation(new Vector3(xPosition, 0, 0));
            activationFlowModels.add(flowInstance);
        }
    }
    
    public void updateFromSnapshot(LayerSnapshot snapshot, float animationTime) {
        this.pulsePhase = animationTime;
        
        // Calculate activation intensity from snapshot
        if (snapshot != null) {
            activationIntensity = snapshot.activationIntensity;
            
            // Update colors based on activation
            Color targetColor = new Color(
                baseColor.r * (1 - activationIntensity) + activeColor.r * activationIntensity,
                baseColor.g * (1 - activationIntensity) + activeColor.g * activationIntensity,
                baseColor.b * (1 - activationIntensity) + activeColor.b * activationIntensity,
                1.0f
            );
            
            // Apply color update to main instance
            if (mainInstance != null && mainInstance.model != null) {
                applyColorToModel(mainInstance.model, targetColor);
            }
            
            // Update neuron colors
            for (ModelInstance neuron : neuronModels) {
                applyColorToModel(neuron.model, targetColor);
            }
        }
    }
    
    private void applyColorToModel(Model model, Color color) {
        if (model == null) return;
        
        // Simplified color update - just recreate with new color for now
        // A full implementation would update existing materials
    }
    
    public void render(ModelBatch modelBatch, Environment environment) {
        // Render main layer body
        if (mainInstance != null) {
            modelBatch.render(mainInstance, environment);
        }
        
        // Render neurons
        for (ModelInstance neuron : neuronModels) {
            modelBatch.render(neuron, environment);
        }
        
        // Render activation flow
        float layerDepth = 4.0f;
        for (int i = 0; i < activationFlowModels.size(); i++) {
            ModelInstance flow = activationFlowModels.get(i);
            
            // Animate flow particles
            float offset = (pulsePhase * 2.0f + i * 0.5f) % 1.0f;
            float z = -layerDepth / 2 + offset * layerDepth;
            
            flow.transform.setTranslation(new Vector3(xPosition, 0, z));
            modelBatch.render(flow, environment);
        }
    }
    
    private float getLayerHeight() {
        if (layerInfo == null) return 2.0f;
        
        // Scale height based on layer size
        return Math.max(2.0f, Math.min(8.0f, layerInfo.outputChannels / 32.0f));
    }
    
    private int getNeuronCount() {
        if (layerInfo == null) return 8;
        
        // Calculate neuron count based on layer type
        switch (layerInfo.type) {
            case INPUT: return 8;
            case CONV2D: return Math.min(16, layerInfo.outputChannels);
            case BILSTM: return Math.min(8, layerInfo.outputChannels / 64);
            case DENSE:
            case OUTPUT: return Math.min(12, layerInfo.outputChannels);
            default: return 8;
        }
    }
    
    private float getLayerSize(NetworkArchitecture.LayerInfo layerInfo) {
        if (layerInfo == null) return 1.0f;
        
        float size = 1.0f;
        
        // Scale size based on output channels
        size = Math.max(0.5f, Math.min(3.0f, layerInfo.outputChannels / 64.0f));
        
        return size;
    }
    
    public void dispose() {
        // Dispose all models
        if (mainModel != null) {
            mainModel.dispose();
        }
        
        for (ModelInstance neuron : neuronModels) {
            if (neuron.model != null) {
                neuron.model.dispose();
            }
        }
        
        for (ModelInstance flow : activationFlowModels) {
            if (flow.model != null) {
                flow.model.dispose();
            }
        }
    }
    
    public String getName() {
        return layerInfo != null ? layerInfo.name : "Unknown";
    }
}
