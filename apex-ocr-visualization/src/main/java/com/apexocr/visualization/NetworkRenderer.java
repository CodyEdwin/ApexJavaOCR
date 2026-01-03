package com.apexocr.visualization;

import com.apexocr.training.monitoring.NetworkArchitecture;
import com.apexocr.training.monitoring.TrainingSnapshot;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.*;
import com.badlogic.gdx.graphics.g3d.*;
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute;
import com.badlogic.gdx.graphics.g3d.environment.DirectionalLight;
import com.badlogic.gdx.graphics.g3d.environment.PointLight;
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder;
import com.badlogic.gdx.math.Matrix4;
import com.badlogic.gdx.math.Vector3;
import com.badlogic.gdx.utils.Array;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Renders the neural network in 3D using LibGDX.
 * Handles 3D model creation, lighting, and camera management.
 */
public class NetworkRenderer {
    
    private final ModelBatch modelBatch;
    private final Environment environment;
    private final Map<String, Layer3D> layerModels = new ConcurrentHashMap<>();
    private final Map<String, Connection3D> connections = new ConcurrentHashMap<>();
    
    private NetworkArchitecture networkArchitecture;
    private TrainingSnapshot latestSnapshot;
    
    private Camera camera;
    private CameraInputController cameraController;
    
    private float animationTime = 0f;
    private float cameraDistance = 20f;
    private float cameraAngleY = 0f;
    private float cameraAngleX = 30f;
    
    // Visual constants
    private static final Color INPUT_COLOR = new Color(0.23f, 0.51f, 0.96f, 1f); // Blue
    private static final Color CONV_COLOR = new Color(0.55f, 0.36f, 0.96f, 1f); // Purple
    private static final Color LSTM_COLOR = new Color(0.06f, 0.71f, 0.51f, 1f); // Green
    private static final Color DENSE_COLOR = new Color(0.93f, 0.27f, 0.27f, 1f); // Red
    private static final Color OUTPUT_COLOR = new Color(0.96f, 0.76f, 0.03f, 1f); // Yellow
    private static final Color ACTIVATION_COLOR = new Color(0.95f, 0.95f, 0.95f, 1f); // White
    private static final Color GRADIENT_COLOR = new Color(1f, 0.5f, 0f, 1f); // Orange
    
    public NetworkRenderer() {
        this.modelBatch = new ModelBatch();
        this.environment = createEnvironment();
        createCamera();
    }
    
    /**
     * Set the network architecture to render.
     * @param architecture Network architecture description
     */
    public void setNetworkArchitecture(NetworkArchitecture architecture) {
        this.networkArchitecture = architecture;
        rebuildNetwork();
    }
    
    /**
     * Update the visual state based on the latest training snapshot.
     * @param snapshot Latest training snapshot
     */
    public void updateFromSnapshot(TrainingSnapshot snapshot) {
        this.latestSnapshot = snapshot;
        updateLayerVisuals(snapshot);
    }
    
    /**
     * Render the network.
     * @param delta Time since last render
     */
    public void render(float delta) {
        animationTime += delta;
        
        updateCamera();
        
        modelBatch.begin(camera);
        modelBatch.render(getAllRenderables(), environment);
        modelBatch.end();
    }
    
    /**
     * Dispose of resources.
     */
    public void dispose() {
        modelBatch.dispose();
        for (Layer3D layer : layerModels.values()) {
            layer.dispose();
        }
        for (Connection3D conn : connections.values()) {
            conn.dispose();
        }
    }
    
    /**
     * Get all renderable objects for the scene.
     * @return Array of renderables
     */
    private Array<Renderable> getAllRenderables() {
        Array<Renderable> renderables = new Array<>();
        
        // Add all layer models
        for (Layer3D layer : layerModels.values()) {
            layer.getRenderables(renderables);
        }
        
        // Add all connection models
        for (Connection3D conn : connections.values()) {
            conn.getRenderables(renderables);
        }
        
        return renderables;
    }
    
    /**
     * Create the 3D environment with lights.
     * @return Configured environment
     */
    private Environment createEnvironment() {
        Environment env = new Environment();
        
        // Ambient light
        env.set(new ColorAttribute(ColorAttribute.AmbientLight, 0.3f, 0.3f, 0.35f, 1f));
        
        // Directional light (sun)
        DirectionalLight sun = new DirectionalLight();
        sun.intensity = 0.8f;
        sun.direction.set(1f, -1f, -0.5f).nor();
        env.add(sun);
        
        // Fill light
        PointLight fillLight = new PointLight();
        fillLight.intensity = 0.3f;
        fillLight.position.set(-10f, 10f, 10f);
        env.add(fillLight);
        
        return env;
    }
    
    /**
     * Create and configure the 3D camera.
     */
    private void createCamera() {
        camera = new PerspectiveCamera(60, Gdx.graphics.getWidth(), Gdx.graphics.getHeight());
        camera.near = 0.1f;
        camera.far = 1000f;
        
        updateCamera();
    }
    
    /**
     * Update camera position based on orbit controls.
     */
    private void updateCamera() {
        float x = (float) (cameraDistance * Math.sin(Math.toRadians(cameraAngleY)) * Math.cos(Math.toRadians(cameraAngleX)));
        float y = (float) (cameraDistance * Math.sin(Math.toRadians(cameraAngleX)));
        float z = (float) (cameraDistance * Math.cos(Math.toRadians(cameraAngleY)) * Math.cos(Math.toRadians(cameraAngleX)));
        
        camera.position.set(x, y, z);
        camera.lookAt(0, 0, 0);
        camera.update();
    }
    
    /**
     * Rebuild the entire network visualization.
     */
    private void rebuildNetwork() {
        if (networkArchitecture == null) return;
        
        // Clear existing models
        for (Layer3D layer : layerModels.values()) layer.dispose();
        for (Connection3D conn : connections.values()) conn.dispose();
        layerModels.clear();
        connections.clear();
        
        // Get layer info list
        var layers = networkArchitecture.layers;
        
        // Create 3D models for each layer
        for (int i = 0; i < layers.size(); i++) {
            NetworkArchitecture.LayerInfo layerInfo = layers.get(i);
            float xPosition = (i - layers.size() / 2f) * 3f;
            
            Layer3D layer = createLayer3D(layerInfo, xPosition);
            layerModels.put(layerInfo.name, layer);
        }
        
        // Create connections between layers
        for (int i = 0; i < layers.size() - 1; i++) {
            NetworkArchitecture.LayerInfo from = layers.get(i);
            NetworkArchitecture.LayerInfo to = layers.get(i + 1);
            
            Connection3D connection = createConnection3D(from, to);
            connections.put(from.name + "_to_" + to.name, connection);
        }
    }
    
    /**
     * Create a 3D representation of a layer.
     * @param layerInfo Layer information
     * @param xPosition X position in 3D space
     * @return Layer3D object
     */
    private Layer3D createLayer3D(NetworkArchitecture.LayerInfo layerInfo, float xPosition) {
        Layer3D layer = new Layer3D(layerInfo.name);
        
        ModelBuilder modelBuilder = new ModelBuilder();
        float size = getLayerSize(layerInfo);
        
        Color color = getLayerColor(layerInfo.type);
        
        switch (layerInfo.type) {
            case INPUT:
                // Input as a cube
                modelBuilder.begin();
                modelBuilder.createBox(size, size, size, 
                    new Material(ColorAttribute.createDiffuse(color)));
                break;
                
            case CONV2D:
                // Conv2D as a flat plate
                modelBuilder.begin();
                modelBuilder.createBox(size * 2, 0.3f, size * 2,
                    new Material(ColorAttribute.createDiffuse(color)));
                break;
                
            case BILSTM:
                // BiLSTM as cylinders/helix
                modelBuilder.begin();
                modelBuilder.createCylinder(size * 0.5f, size, size, 16,
                    new Material(ColorAttribute.createDiffuse(color)));
                break;
                
            case DENSE:
                // Dense as a sphere
                modelBuilder.begin();
                modelBuilder.createSphere(size, size, size, 16, 16,
                    new Material(ColorAttribute.createDiffuse(color)));
                break;
                
            case OUTPUT:
                // Output as a cone/pyramid
                modelBuilder.begin();
                modelBuilder.createCone(size * 0.8f, size, size * 0.8f, 8,
                    new Material(ColorAttribute.createDiffuse(color)));
                break;
                
            default:
                // Default cube
                modelBuilder.begin();
                modelBuilder.createBox(size, size, size,
                    new Material(ColorAttribute.createDiffuse(color)));
                break;
        }
        
        layer.setModel(modelBuilder.end());
        layer.setPosition(xPosition, 0, 0);
        
        return layer;
    }
    
    /**
     * Create a 3D connection between two layers.
     * @param from Source layer
     * @param to Target layer
     * @return Connection3D object
     */
    private Connection3D createConnection3D(NetworkArchitecture.LayerInfo from, NetworkArchitecture.LayerInfo to) {
        Connection3D connection = new Connection3D(from.name + "_to_" + to.name);
        
        ModelBuilder modelBuilder = new ModelBuilder();
        
        // Create connecting lines as thin cylinders
        float fromX = (networkArchitecture.getLayerIndex(from.name) - networkArchitecture.getLayerCount() / 2f) * 3f;
        float toX = (networkArchitecture.getLayerIndex(to.name) - networkArchitecture.getLayerCount() / 2f) * 3f;
        float distance = toX - fromX;
        
        // Create a tube connecting the layers
        modelBuilder.begin();
        modelBuilder.createCylinder(0.1f, distance, 0.1f, 8,
            new Material(ColorAttribute.createDiffuse(new Color(0.5f, 0.5f, 0.5f, 0.3f))));
        
        connection.setModel(modelBuilder.end());
        connection.setPosition(fromX + distance / 2f, 0, 0);
        
        return connection;
    }
    
    /**
     * Update layer visuals based on training snapshot.
     * @param snapshot Training snapshot
     */
    private void updateLayerVisuals(TrainingSnapshot snapshot) {
        if (snapshot == null) return;
        
        animationTime += Gdx.graphics.getDeltaTime();
        
        for (var entry : snapshot.layerSnapshots.entrySet()) {
            Layer3D layer = layerModels.get(entry.getKey());
            if (layer != null) {
                var layerSnapshot = entry.getValue();
                
                // Update layer intensity based on activation
                float intensity = layerSnapshot.activationIntensity;
                
                // Pulse animation for active layers
                float pulse = (float) (Math.sin(animationTime * 3) * 0.1 + 0.9);
                
                // Scale layer based on gradient flow
                float scale = 1f + layerSnapshot.gradientFlow * 0.2f;
                layer.setScale(scale * pulse);
                
                // Update color based on gradient flow
                float gradientIntensity = layerSnapshot.gradientFlow;
                if (gradientIntensity > 0) {
                    Color baseColor = getLayerColor(layerSnapshot.layerType);
                    Color gradientColor = new Color(
                        baseColor.r + (GRADIENT_COLOR.r - baseColor.r) * gradientIntensity,
                        baseColor.g + (GRADIENT_COLOR.g - baseColor.g) * gradientIntensity,
                        baseColor.b + (GRADIENT_COLOR.b - baseColor.b) * gradientIntensity,
                        1f
                    );
                    layer.setColor(gradientColor);
                }
            }
        }
    }
    
    /**
     * Get the visual size for a layer type.
     * @param layerInfo Layer information
     * @return Size value
     */
    private float getLayerSize(NetworkArchitecture.LayerInfo layerInfo) {
        switch (layerInfo.type) {
            case INPUT:
                return 1.5f;
            case CONV2D:
                return 2f;
            case BILSTM:
                return 1.8f;
            case DENSE:
                return 1.2f;
            case OUTPUT:
                return 1f;
            default:
                return 1f;
        }
    }
    
    /**
     * Get the color for a layer type.
     * @param type Layer type
     * @return Color for the layer
     */
    private Color getLayerColor(com.apexocr.training.monitoring.LayerSnapshot.LayerType type) {
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
                return ACTIVATION_COLOR;
        }
    }
    
    /**
     * Set camera distance.
     * @param distance Camera distance
     */
    public void setCameraDistance(float distance) {
        this.cameraDistance = Math.max(5f, Math.min(100f, distance));
    }
    
    /**
     * Set camera rotation angles.
     * @param angleY Y rotation angle in degrees
     * @param angleX X rotation angle in degrees
     */
    public void setCameraAngles(float angleY, float angleX) {
        this.cameraAngleY = angleY;
        this.cameraAngleX = Math.max(-89f, Math.min(89f, angleX));
    }
    
    /**
     * Reset camera to default position.
     */
    public void resetCamera() {
        cameraDistance = 20f;
        cameraAngleY = 0f;
        cameraAngleX = 30f;
    }
    
    /**
     * Get the camera for input handling.
     * @return The 3D camera
     */
    public Camera getCamera() {
        return camera;
    }
}
