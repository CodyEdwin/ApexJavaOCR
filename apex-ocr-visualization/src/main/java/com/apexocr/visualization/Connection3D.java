package com.apexocr.visualization;

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
 * Represents connections between two neural network layers in 3D space.
 * Visualizes weight connections with animated flow.
 */
public class Connection3D {
    
    private final NetworkArchitecture.LayerInfo fromLayer;
    private final NetworkArchitecture.LayerInfo toLayer;
    
    // 3D components
    private List<ModelInstance> connectionModels;
    private List<ModelInstance> flowModels;
    
    // Visual properties
    private Color connectionColor;
    private Color activeColor;
    private float animationPhase;
    
    // Connection parameters
    private final float fromX;
    private final float toX;
    
    public Connection3D(NetworkArchitecture.LayerInfo fromLayer, NetworkArchitecture.LayerInfo toLayer) {
        this.fromLayer = fromLayer;
        this.toLayer = toLayer;
        
        this.fromX = fromLayer.position[0];
        this.toX = toLayer.position[0];
        
        this.connectionColor = new Color(0.3f, 0.3f, 0.4f, 0.5f);
        this.activeColor = new Color(0.5f, 0.8f, 1.0f, 0.8f);
        this.animationPhase = 0f;
        
        this.connectionModels = new ArrayList<>();
        this.flowModels = new ArrayList<>();
        
        // Build 3D representation
        build3DRepresentation();
    }
    
    private void build3DRepresentation() {
        ModelBuilder modelBuilder = new ModelBuilder();
        
        // Create main connection lines
        createConnectionLines(modelBuilder);
        
        // Create flow particles
        createFlowParticles(modelBuilder);
    }
    
    private void createConnectionLines(ModelBuilder modelBuilder) {
        int connectionCount = getConnectionCount();
        float yStart = -2.0f;
        float yStep = 4.0f / (connectionCount + 1);
        float distance = toX - fromX;
        
        Material connectionMaterial = new Material();
        connectionMaterial.set(new ColorAttribute(ColorAttribute.Diffuse, connectionColor));
        
        for (int i = 0; i < connectionCount; i++) {
            float y = yStart + (i + 1) * yStep;
            
            // Create a thin cylinder representing the connection
            Model connectionModel = modelBuilder.createCylinder(0.05f, distance, 0.05f, 4, connectionMaterial,
                VertexAttributes.Usage.Position | VertexAttributes.Usage.Normal);
            
            ModelInstance connectionInstance = new ModelInstance(connectionModel);
            
            // Position and rotate the connection
            connectionInstance.transform.setTranslation(new Vector3(
                (fromX + toX) / 2, // Midpoint X
                y,                 // Y position
                0                  // Z position
            ));
            
            // Rotate 90 degrees around Z axis to align with X axis
            connectionInstance.transform.rotate(new Vector3(0, 0, 1), 90);
            
            connectionModels.add(connectionInstance);
        }
    }
    
    private void createFlowParticles(ModelBuilder modelBuilder) {
        int flowParticleCount = 20;
        
        Material flowMaterial = new Material();
        flowMaterial.set(new ColorAttribute(ColorAttribute.Emissive, activeColor));
        
        for (int i = 0; i < flowParticleCount; i++) {
            float y = (float) (Math.random() * 4.0 - 2.0);
            float z = (float) (Math.random() * 2.0 - 1.0);
            
            Model flowModel = modelBuilder.createSphere(0.1f, 0.1f, 0.1f, 6, 6, flowMaterial,
                VertexAttributes.Usage.Position | VertexAttributes.Usage.Normal);
            
            ModelInstance flowInstance = new ModelInstance(flowModel);
            
            // Initialize at starting position
            flowInstance.transform.setTranslation(new Vector3(fromX, y, z));
            
            flowModels.add(flowInstance);
        }
    }
    
    public void render(ModelBatch modelBatch, Environment environment, float animationTime) {
        this.animationPhase = animationTime;
        
        // Render connection lines
        for (ModelInstance connection : connectionModels) {
            modelBatch.render(connection, environment);
        }
        
        // Update and render flow particles
        float distance = toX - fromX;
        
        for (ModelInstance flow : flowModels) {
            // Animate flow particles along the connection
            float progress = (animationPhase * 0.5f + flowModels.indexOf(flow) * 0.05f) % 1.0f;
            float x = fromX + progress * distance;
            
            // Get original Y and Z positions
            Vector3 pos = new Vector3();
            flow.transform.getTranslation(pos);
            
            // Update position with animation
            flow.transform.setTranslation(new Vector3(x, pos.y, pos.z));
            
            modelBatch.render(flow, environment);
        }
    }
    
    private int getConnectionCount() {
        // Determine number of connection lines based on layer sizes
        int fromSize = getLayerRepresentationSize(fromLayer);
        int toSize = getLayerRepresentationSize(toLayer);
        
        // Use a reasonable number for visualization
        return Math.min(20, Math.max(fromSize, toSize));
    }
    
    private int getLayerRepresentationSize(NetworkArchitecture.LayerInfo layer) {
        if (layer == null) return 8;
        
        switch (layer.type) {
            case INPUT: return 8;
            case CONV2D: return Math.min(16, layer.outputChannels);
            case BILSTM: return Math.min(8, layer.outputChannels / 64);
            case DENSE:
            case OUTPUT: return Math.min(12, layer.outputChannels);
            default: return 8;
        }
    }
    
    public void dispose() {
        // Dispose all models
        for (ModelInstance connection : connectionModels) {
            if (connection.model != null) {
                connection.model.dispose();
            }
        }
        
        for (ModelInstance flow : flowModels) {
            if (flow.model != null) {
                flow.model.dispose();
            }
        }
    }
    
    public String getFromLayerName() {
        return fromLayer != null ? fromLayer.name : "Unknown";
    }
    
    public String getToLayerName() {
        return toLayer != null ? toLayer.name : "Unknown";
    }
}
