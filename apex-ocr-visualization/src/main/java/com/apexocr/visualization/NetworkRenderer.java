package com.apexocr.visualization;

import com.apexocr.core.monitoring.NetworkArchitecture;
import com.apexocr.core.monitoring.TrainingSnapshot;
import com.apexocr.core.monitoring.LayerSnapshot;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g3d.Environment;
import com.badlogic.gdx.graphics.g3d.Material;
import com.badlogic.gdx.graphics.g3d.Model;
import com.badlogic.gdx.graphics.g3d.ModelBatch;
import com.badlogic.gdx.graphics.g3d.ModelInstance;
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute;
import com.badlogic.gdx.graphics.g3d.environment.DirectionalLight;
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder;
import com.badlogic.gdx.graphics.VertexAttributes;
import com.badlogic.gdx.graphics.g3d.utils.CameraInputController;
import com.badlogic.gdx.math.Vector3;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Renders the neural network in 3D space.
 * Manages layer nodes and connection lines with real-time updates.
 */
public class NetworkRenderer {
    
    private NetworkArchitecture networkArchitecture;
    private TrainingSnapshot latestSnapshot;
    
    // 3D rendering components
    private ModelBatch modelBatch;
    private Environment environment;
    private CameraInputController cameraController;
    
    // 3D objects
    private Map<String, Layer3D> layer3DObjects;
    private Map<String, Connection3D> connection3DObjects;
    private List<ModelInstance> decorativeModels;
    
    // Animation
    private float animationTime;
    private float[] layerPulsePhases;
    
    // Colors for different layer types
    private static final Color INPUT_COLOR = new Color(0.2f, 0.6f, 1.0f, 1.0f);
    private static final Color CONV_COLOR = new Color(0.2f, 0.8f, 0.4f, 1.0f);
    private static final Color LSTM_COLOR = new Color(0.8f, 0.4f, 0.8f, 1.0f);
    private static final Color DENSE_COLOR = new Color(1.0f, 0.6f, 0.2f, 1.0f);
    private static final Color OUTPUT_COLOR = new Color(0.2f, 1.0f, 0.4f, 1.0f);
    private static final Color DEFAULT_COLOR = new Color(0.6f, 0.6f, 0.6f, 1.0f);
    
    public NetworkRenderer() {
        initializeRendering();
        initializeObjects();
    }
    
    private void initializeRendering() {
        // Create model batch for rendering
        modelBatch = new ModelBatch();
        
        // Create environment with lighting
        environment = new Environment();
        environment.set(new ColorAttribute(ColorAttribute.AmbientLight, 0.4f, 0.4f, 0.4f, 1.0f));
        
        DirectionalLight light = new DirectionalLight();
        light.direction.set(1.0f, 1.0f, 1.0f).nor();
        light.color.set(1.0f, 1.0f, 1.0f, 1.0f);
        environment.add(light);
        
        // Add backlight
        DirectionalLight backlight = new DirectionalLight();
        backlight.direction.set(-1.0f, -0.5f, -1.0f).nor();
        backlight.color.set(0.3f, 0.3f, 0.5f, 1.0f);
        environment.add(backlight);
        
        // Create camera controller for user interaction
        cameraController = new CameraInputController(null);
        Gdx.input.setInputProcessor(cameraController);
    }
    
    private void initializeObjects() {
        layer3DObjects = new HashMap<>();
        connection3DObjects = new HashMap<>();
        decorativeModels = new ArrayList<>();
        animationTime = 0f;
        layerPulsePhases = new float[0];
    }
    
    public void setNetworkArchitecture(NetworkArchitecture architecture) {
        this.networkArchitecture = architecture;
        this.layerPulsePhases = new float[architecture.getLayerCount()];
        
        // Clear existing objects
        layer3DObjects.clear();
        connection3DObjects.clear();
        decorativeModels.clear();
        
        // Build 3D representation
        buildNetwork3D();
    }
    
    private void buildNetwork3D() {
        if (networkArchitecture == null || networkArchitecture.layers == null) {
            return;
        }
        
        List<NetworkArchitecture.LayerInfo> layers = networkArchitecture.layers;
        
        // Create Layer3D objects for each layer
        for (int i = 0; i < layers.size(); i++) {
            NetworkArchitecture.LayerInfo layerInfo = layers.get(i);
            float xPosition = i * 12.0f; // Spacing between layers
            
            Layer3D layer3D = new Layer3D(layerInfo, xPosition);
            layer3DObjects.put(layerInfo.name, layer3D);
        }
        
        // Create connections between adjacent layers
        for (int i = 0; i < layers.size() - 1; i++) {
            NetworkArchitecture.LayerInfo from = layers.get(i);
            NetworkArchitecture.LayerInfo to = layers.get(i + 1);
            
            Connection3D connection = new Connection3D(from, to);
            connection3DObjects.put(from.name + "_to_" + to.name, connection);
        }
        
        // Create decorative elements
        createDecorativeElements(layers);
    }
    
    private void createDecorativeElements(List<NetworkArchitecture.LayerInfo> layers) {
        // Create floor grid
        createFloorGrid();
        
        // Create axis indicators
        createAxisIndicators();
    }
    
    private void createFloorGrid() {
        ModelBuilder modelBuilder = new ModelBuilder();
        
        // Create a simple grid on the XZ plane
        int gridSize = 20;
        float gridSpacing = 2.0f;
        
        Material gridMaterial = new Material();
        gridMaterial.set(new ColorAttribute(ColorAttribute.Diffuse, 0.15f, 0.15f, 0.2f, 1.0f));
        
        Model gridModel = modelBuilder.createBox(gridSize * gridSpacing, 0.1f, gridSize * gridSpacing,
            gridMaterial, VertexAttributes.Usage.Position | VertexAttributes.Usage.Normal);
        
        ModelInstance gridInstance = new ModelInstance(gridModel);
        gridInstance.transform.setTranslation(new Vector3(30, -5, 0));
        decorativeModels.add(gridInstance);
    }
    
    private void createAxisIndicators() {
        // Create simple axis indicators
        ModelBuilder modelBuilder = new ModelBuilder();
        
        // X axis (red)
        Material xMaterial = new Material();
        xMaterial.set(new ColorAttribute(ColorAttribute.Diffuse, 1.0f, 0.2f, 0.2f, 1.0f));
        Model xAxis = modelBuilder.createCylinder(0.1f, 5f, 0.1f, 8, xMaterial,
            VertexAttributes.Usage.Position | VertexAttributes.Usage.Normal);
        decorativeModels.add(new ModelInstance(xAxis));
        
        // Y axis (green)
        Material yMaterial = new Material();
        yMaterial.set(new ColorAttribute(ColorAttribute.Diffuse, 0.2f, 1.0f, 0.2f, 1.0f));
        Model yAxis = modelBuilder.createCylinder(0.1f, 5f, 0.1f, 8, yMaterial,
            VertexAttributes.Usage.Position | VertexAttributes.Usage.Normal);
        decorativeModels.add(new ModelInstance(yAxis));
        
        // Z axis (blue)
        Material zMaterial = new Material();
        zMaterial.set(new ColorAttribute(ColorAttribute.Diffuse, 0.2f, 0.2f, 1.0f, 1.0f));
        Model zAxis = modelBuilder.createCylinder(0.1f, 5f, 0.1f, 8, zMaterial,
            VertexAttributes.Usage.Position | VertexAttributes.Usage.Normal);
        decorativeModels.add(new ModelInstance(zAxis));
    }
    
    public void updateFromSnapshot(TrainingSnapshot snapshot) {
        this.latestSnapshot = snapshot;
        
        // Update layer visualizations based on snapshot
        if (snapshot.layerSnapshots != null) {
            for (Map.Entry<String, LayerSnapshot> entry : snapshot.layerSnapshots.entrySet()) {
                String layerName = entry.getKey();
                LayerSnapshot layerSnapshot = entry.getValue();
                
                Layer3D layer3D = layer3DObjects.get(layerName);
                if (layer3D != null) {
                    layer3D.updateFromSnapshot(layerSnapshot, animationTime);
                }
            }
        }
    }
    
    public void render(float delta) {
        animationTime += delta;
        
        // Update camera
        cameraController.update();
        
        // Begin rendering
        modelBatch.begin(cameraController.camera);
        
        // Render decorative models
        for (ModelInstance decorativeModel : decorativeModels) {
            modelBatch.render(decorativeModel, environment);
        }
        
        // Render layer 3D objects
        for (Layer3D layer3D : layer3DObjects.values()) {
            layer3D.render(modelBatch, environment);
        }
        
        // Render connection 3D objects
        for (Connection3D connection3D : connection3DObjects.values()) {
            connection3D.render(modelBatch, environment, animationTime);
        }
        
        // End rendering
        modelBatch.end();
    }
    
    public void resize(int width, int height) {
        if (cameraController != null && cameraController.camera != null) {
            cameraController.camera.viewportWidth = width;
            cameraController.camera.viewportHeight = height;
            cameraController.camera.update();
        }
    }
    
    public void dispose() {
        // Dispose all 3D objects
        for (Layer3D layer3D : layer3DObjects.values()) {
            layer3D.dispose();
        }
        
        for (Connection3D connection3D : connection3DObjects.values()) {
            connection3D.dispose();
        }
        
        for (ModelInstance decorativeModel : decorativeModels) {
            decorativeModel.model.dispose();
        }
        
        // Dispose rendering resources
        modelBatch.dispose();
    }
    
    /**
     * Get the color for a specific layer type.
     */
    public static Color getLayerColor(LayerSnapshot.LayerType type) {
        if (type == null) return DEFAULT_COLOR;
        
        switch (type) {
            case INPUT:
                return INPUT_COLOR;
            case CONV2D:
                return CONV_COLOR;
            case BILSTM:
                return LSTM_COLOR;
            case DENSE:
                return DENSE_COLOR;
            case OUTPUT:
                return OUTPUT_COLOR;
            default:
                return DEFAULT_COLOR;
        }
    }
    
    /**
     * Get a short string representation of layer type.
     */
    public static String getLayerTypeShort(LayerSnapshot.LayerType type) {
        if (type == null) return "?";
        switch (type) {
            case INPUT: return "IN";
            case CONV2D: return "CONV";
            case BILSTM: return "LSTM";
            case DENSE: return "FC";
            case OUTPUT: return "OUT";
            default: return type.name().substring(0, Math.min(3, type.name().length()));
        }
    }
}
