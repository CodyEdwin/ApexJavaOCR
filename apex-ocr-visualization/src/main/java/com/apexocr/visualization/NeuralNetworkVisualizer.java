package com.apexocr.visualization;

import com.apexocr.core.tensor.Tensor;
import com.apexocr.core.neural.Layer;
import com.apexocr.core.neural.Conv2D;
import com.apexocr.core.neural.Dense;
import com.apexocr.core.neural.BiLSTM;
import com.apexocr.engine.OcrEngine;
import com.badlogic.gdx.ApplicationAdapter;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Graphics;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;
import com.badlogic.gdx.graphics.*;
import com.badlogic.gdx.graphics.g3d.*;
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute;
import com.badlogic.gdx.graphics.g3d.environment.DirectionalLight;
import com.badlogic.gdx.graphics.g3d.environment.PointLight;
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder;
import com.badlogic.gdx.math.Matrix4;
import com.badlogic.gdx.math.Vector3;
import com.badlogic.gdx.utils.Array;
import com.badlogic.gdx.utils.Disposable;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * NeuralNetworkVisualizer - 3D visualization of the OCR neural network.
 * 
 * Features:
 * - 3D representation of the CRNN architecture (Conv2D → BiLSTM → Dense)
 * - Real-time animation of forward passes and backpropagation
 * - Interactive camera controls (orbit, zoom, pan)
 * - Live metrics display (loss, accuracy, epoch)
 * - Opens automatically when neural network is active
 *
 * @author ApexOCR Team
 */
public class NeuralNetworkVisualizer extends ApplicationAdapter {
    
    // ==================== STATIC ACCESS FOR EXTERNAL CONTROL ====================
    
    private static NeuralNetworkVisualizer instance;
    private static boolean headlessMode = false;
    
    /**
     * Gets the singleton instance of the visualizer.
     */
    public static NeuralNetworkVisualizer getInstance() {
        return instance;
    }
    
    /**
     * Sets whether to run in headless mode (no window).
     */
    public static void setHeadlessMode(boolean headless) {
        headlessMode = headless;
    }
    
    /**
     * Opens the 3D visualization window.
     * Call this when starting neural network operations.
     */
    public static void open(OcrEngine engine) {
        if (headlessMode) return;
        if (instance != null) {
            instance.setEngine(engine);
            return;
        }
        
        Lwjgl3ApplicationConfiguration config = new Lwjgl3ApplicationConfiguration();
        config.setTitle("Neural Network Visualizer - ApexOCR");
        config.setWindowSizeLimits(1200, 800, 1920, 1080);
        config.setResizable(true);
        config.useVsync(true);
        config.setForegroundFPS(60);
        
        instance = new NeuralNetworkVisualizer(engine);
        new Lwjgl3Application(instance, config);
    }
    
    /**
     * Opens the visualizer with default configuration.
     */
    public static void open() {
        open(null);
    }
    
    /**
     * Closes the visualization window.
     */
    public static void close() {
        if (instance != null) {
            instance.dispose();
            instance = null;
        }
    }
    
    // ==================== INSTANCE VARIABLES ====================
    
    private OcrEngine engine;
    private boolean isActive = false;
    
    // 3D Rendering
    private PerspectiveCamera camera;
    private ModelBatch modelBatch;
    private Environment environment;
    private Array<ModelInstance> visibleModels;
    
    // Network Components
    private List<LayerVisualizer> layerVisualizers;
    private Map<String, NeuronGroup> neuronGroups;
    private List<ConnectionGroup> connectionGroups;
    
    // Animation
    private List<DataPacket> activePackets;
    private List<GradientPacket> gradientPackets;
    private float animationSpeed = 1.0f;
    private float timeElapsed = 0;
    
    // Camera Controls
    private OrbitCameraController cameraController;
    private float cameraDistance = 20f;
    private float cameraAngleX = 45f;
    private float cameraAngleY = 30f;
    
    // Metrics
    private float currentLoss = 0;
    private float currentAccuracy = 0;
    private int currentEpoch = 0;
    private int currentStep = 0;
    private String currentPhase = "Idle";
    
    // Event Queue for thread-safe updates
    private ConcurrentLinkedQueue<VisualizationEvent> eventQueue;
    
    // Colors
    private static final Color COLOR_CONV2D = new Color(0x00FFFFFF); // Cyan
    private static final Color COLOR_BILSTM = new Color(0x9D00FFFF); // Purple
    private static final Color COLOR_DENSE = new Color(0xFFA500FF);  // Orange
    private static final Color COLOR_INPUT = new Color(0x00FF00FF);  // Green
    private static final Color COLOR_OUTPUT = new Color(0xFFFF00FF); // Yellow
    private static final Color COLOR_INACTIVE = new Color(0x333333FF); // Dark gray
    
    // ==================== CONSTRUCTOR ====================
    
    public NeuralNetworkVisualizer(OcrEngine engine) {
        this.engine = engine;
        this.layerVisualizers = new ArrayList<>();
        this.neuronGroups = new HashMap<>();
        this.connectionGroups = new ArrayList<>();
        this.activePackets = new ArrayList<>();
        this.gradientPackets = new ArrayList<>();
        this.eventQueue = new ConcurrentLinkedQueue<>();
    }
    
    public void setEngine(OcrEngine engine) {
        this.engine = engine;
        if (engine != null) {
            buildNetworkVisualization();
        }
    }
    
    // ==================== LIFECYCLE METHODS ====================
    
    @Override
    public void create() {
        // Initialize 3D rendering
        initializeCamera();
        initializeEnvironment();
        initializeModels();
        
        // Set up camera controller
        cameraController = new OrbitCameraController(camera);
        Gdx.input.setInputProcessor(cameraController);
        
        // Start visualization if engine is available
        if (engine != null) {
            buildNetworkVisualization();
            isActive = true;
        }
        
        System.out.println("[NeuralNetworkVisualizer] 3D visualization initialized");
    }
    
    @Override
    public void render() {
        // Process events from other threads
        processEvents();
        
        // Update animation time
        timeElapsed += Gdx.graphics.getDeltaTime() * animationSpeed;
        
        // Clear screen
        Gdx.gl.glClearColor(0.05f, 0.05f, 0.08f, 1f);
        Gdx.gl.glClear(GL30.GL_COLOR_BUFFER_BIT | GL30.GL_DEPTH_BUFFER_BIT);
        
        // Update camera
        cameraController.update();
        camera.update();
        
        // Begin rendering
        modelBatch.begin(camera);
        
        // Render all model instances
        for (ModelInstance model : visibleModels) {
            modelBatch.render(model, environment);
        }
        
        // Render neurons with glow effects
        for (LayerVisualizer layer : layerVisualizers) {
            layer.render(modelBatch, environment, timeElapsed);
        }
        
        // Render connections
        for (ConnectionGroup connections : connectionGroups) {
            connections.render(modelBatch, environment);
        }
        
        // Render data flow packets
        renderDataPackets();
        
        // Render gradient packets
        renderGradientPackets();
        
        modelBatch.end();
        
        // Render 2D overlay
        renderOverlay();
        
        // Check for engine updates if active
        if (isActive && engine != null) {
            pollEngineState();
        }
    }
    
    @Override
    public void resize(int width, int height) {
        camera.viewportWidth = width;
        camera.viewportHeight = height;
        camera.update();
    }
    
    @Override
    public void dispose() {
        // Dispose all resources
        for (ModelInstance model : visibleModels) {
            model.model.dispose();
        }
        if (modelBatch != null) {
            modelBatch.dispose();
        }
        for (LayerVisualizer layer : layerVisualizers) {
            layer.dispose();
        }
        for (ConnectionGroup connections : connectionGroups) {
            connections.dispose();
        }
        for (DataPacket packet : activePackets) {
            packet.dispose();
        }
        for (GradientPacket packet : gradientPackets) {
            packet.dispose();
        }
        
        System.out.println("[NeuralNetworkVisualizer] Resources disposed");
    }
    
    // ==================== INITIALIZATION ====================
    
    private void initializeCamera() {
        camera = new PerspectiveCamera(60, Gdx.graphics.getWidth(), Gdx.graphics.getHeight());
        camera.position.set(0, 5, cameraDistance);
        camera.lookAt(0, 0, 0);
        camera.near = 0.1f;
        camera.far = 1000f;
        camera.update();
    }
    
    private void initializeEnvironment() {
        environment = new Environment();
        
        // Ambient light
        environment.set(new ColorAttribute(ColorAttribute.AmbientLight, 0.3f, 0.3f, 0.3f, 1f));
        
        // Main directional light
        DirectionalLight dirLight = new DirectionalLight();
        dirLight.direction.set(1, -1, 1).nor();
        dirLight.intensity = 0.8f;
        environment.add(dirLight);
        
        // Fill light
        DirectionalLight fillLight = new DirectionalLight();
        fillLight.direction.set(-1, 0, -1).nor();
        fillLight.intensity = 0.3f;
        environment.add(fillLight);
    }
    
    private void initializeModels() {
        modelBatch = new ModelBatch();
        visibleModels = new Array<>();
    }
    
    // ==================== NETWORK VISUALIZATION ====================
    
    /**
     * Builds the 3D visualization of the neural network from the engine's architecture.
     */
    public void buildNetworkVisualization() {
        // Clear existing
        layerVisualizers.clear();
        neuronGroups.clear();
        connectionGroups.clear();
        
        if (engine == null || engine.getNetwork() == null) {
            System.out.println("[NeuralNetworkVisualizer] No network to visualize");
            return;
        }
        
        List<Layer> network = engine.getNetwork();
        
        // Build visualization layer by layer
        float zPosition = 0;
        float layerSpacing = 8f;
        
        for (int i = 0; i < network.size(); i++) {
            Layer layer = network.get(i);
            LayerVisualizer visualizer = createLayerVisualizer(layer, i, zPosition);
            
            if (visualizer != null) {
                layerVisualizers.add(visualizer);
                zPosition += layerSpacing;
            }
        }
        
        // Create connections between layers
        createConnections();
        
        // Add a floor grid
        createFloorGrid();
        
        System.out.println("[NeuralNetworkVisualizer] Built visualization with " + 
            layerVisualizers.size() + " layers");
    }
    
    /**
     * Creates a visualizer for a single layer.
     */
    private LayerVisualizer createLayerVisualizer(Layer layer, int layerIndex, float zPosition) {
        String layerName = layer.getName();
        LayerType type = determineLayerType(layer);
        
        NeuronGroup group = new NeuronGroup();
        group.layerName = layerName;
        group.layerType = type;
        group.zPosition = zPosition;
        
        int neuronCount;
        float neuronSpacing = 0.8f;
        float layerWidth, layerHeight;
        
        switch (type) {
            case CONV2D:
                // Conv2D layers: Grid arrangement
                neuronCount = 64; // Representative sample
                int gridSize = (int) Math.ceil(Math.sqrt(neuronCount));
                layerWidth = gridSize * neuronSpacing;
                layerHeight = gridSize * neuronSpacing;
                
                for (int i = 0; i < neuronCount; i++) {
                    int row = i / gridSize;
                    int col = i % gridSize;
                    float x = (col - gridSize / 2f) * neuronSpacing;
                    float y = (row - gridSize / 2f) * neuronSpacing;
                    group.positions.add(new Vector3(x, y, zPosition));
                }
                break;
                
            case BILSTM:
                // BiLSTM: Linear arrangement with depth
                neuronCount = 256;
                layerWidth = neuronCount * neuronSpacing * 0.5f;
                layerHeight = 2 * neuronSpacing;
                
                for (int i = 0; i < neuronCount; i++) {
                    float x = (i - neuronCount / 2f) * neuronSpacing * 0.5f;
                    group.positions.add(new Vector3(x, 0, zPosition));
                }
                break;
                
            case DENSE:
                // Dense: Compact grid
                neuronCount = 37; // 36 classes + blank for CTC
                int denseGridSize = (int) Math.ceil(Math.sqrt(neuronCount));
                layerWidth = denseGridSize * neuronSpacing;
                layerHeight = denseGridSize * neuronSpacing;
                
                for (int i = 0; i < neuronCount; i++) {
                    int row = i / denseGridSize;
                    int col = i % denseGridSize;
                    float x = (col - denseGridSize / 2f) * neuronSpacing;
                    float y = (row - denseGridSize / 2f) * neuronSpacing;
                    group.positions.add(new Vector3(x, y, zPosition));
                }
                break;
                
            default:
                return null;
        }
        
        group.color = getColorForType(type);
        
        // Create visualizer
        LayerVisualizer visualizer = new LayerVisualizer(group);
        neuronGroups.put(layerName, group);
        
        return visualizer;
    }
    
    /**
     * Determines the type of layer for visualization purposes.
     */
    private LayerType determineLayerType(Layer layer) {
        if (layer instanceof Conv2D) return LayerType.CONV2D;
        if (layer instanceof BiLSTM) return LayerType.BILSTM;
        if (layer instanceof Dense) return LayerType.DENSE;
        return LayerType.UNKNOWN;
    }
    
    /**
     * Gets the color for a layer type.
     */
    private Color getColorForType(LayerType type) {
        switch (type) {
            case CONV2D: return COLOR_CONV2D;
            case BILSTM: return COLOR_BILSTM;
            case DENSE: return COLOR_DENSE;
            default: return COLOR_INACTIVE;
        }
    }
    
    /**
     * Creates visual connections between adjacent layers.
     */
    private void createConnections() {
        if (layerVisualizers.size() < 2) return;
        
        for (int i = 0; i < layerVisualizers.size() - 1; i++) {
            NeuronGroup source = neuronGroups.get(layerVisualizers.get(i).group.layerName);
            NeuronGroup target = neuronGroups.get(layerVisualizers.get(i + 1).group.layerName);
            
            if (source != null && target != null) {
                ConnectionGroup connections = new ConnectionGroup(source, target);
                connectionGroups.add(connections);
            }
        }
    }
    
    /**
     * Creates a floor grid for depth reference.
     */
    private void createFloorGrid() {
        ModelBuilder builder = new ModelBuilder();
        
        // Create grid lines
        builder.begin();
        int gridSize = 20;
        float spacing = 1f;
        
        for (int i = -gridSize; i <= gridSize; i++) {
            // X-axis lines
            builder.addLine(
                new Vector3(i * spacing, -5, -gridSize * spacing),
                new Vector3(i * spacing, -5, gridSize * spacing),
                new Color(0.2f, 0.2f, 0.25f, 0.5f)
            );
            // Z-axis lines
            builder.addLine(
                new Vector3(-gridSize * spacing, -5, i * spacing),
                new Vector3(gridSize * spacing, -5, i * spacing),
                new Color(0.2f, 0.2f, 0.25f, 0.5f)
            );
        }
        
        Model gridModel = builder.end();
        ModelInstance gridInstance = new ModelInstance(gridModel);
        visibleModels.add(gridInstance);
    }
    
    // ==================== ANIMATION ====================
    
    /**
     * Animates a forward pass through the network.
     */
    public void animateForwardPass(float[] inputActivations, float[] outputActivations) {
        eventQueue.offer(new VisualizationEvent(EventType.FORWARD_PASS, 
            new float[][]{inputActivations, outputActivations}));
    }
    
    /**
     * Animates a backward pass (backpropagation).
     */
    public void animateBackwardPass(float[] gradients) {
        eventQueue.offer(new VisualizationEvent(EventType.BACKWARD_PASS, 
            new float[][]{gradients}));
    }
    
    /**
     * Updates metrics display.
     */
    public void updateMetrics(float loss, float accuracy, int epoch, int step, String phase) {
        eventQueue.offer(new VisualizationEvent(EventType.METRICS_UPDATE, 
            new Object[]{loss, accuracy, epoch, step, phase}));
    }
    
    /**
     * Highlights a specific layer during processing.
     */
    public void highlightLayer(String layerName, float intensity) {
        eventQueue.offer(new VisualizationEvent(EventType.LAYER_HIGHLIGHT, 
            new Object[]{layerName, intensity}));
    }
    
    /**
     * Spawns data flow particles.
     */
    private void spawnDataPackets(String fromLayer, String toLayer, float intensity) {
        NeuronGroup source = neuronGroups.get(fromLayer);
        NeuronGroup target = neuronGroups.get(toLayer);
        
        if (source == null || target == null) return;
        
        // Create packets from a sample of source neurons
        int packetCount = Math.min(10, source.positions.size());
        Random random = new Random();
        
        for (int i = 0; i < packetCount; i++) {
            int srcIdx = random.nextInt(source.positions.size());
            int dstIdx = random.nextInt(target.positions.size());
            
            DataPacket packet = new DataPacket(
                source.positions.get(srcIdx),
                target.positions.get(dstIdx),
                intensity,
                COLOR_INPUT
            );
            activePackets.add(packet);
        }
    }
    
    /**
     * Spawns gradient flow particles (for backprop).
     */
    private void spawnGradientPackets(String fromLayer, String toLayer, float intensity) {
        NeuronGroup source = neuronGroups.get(fromLayer);
        NeuronGroup target = neuronGroups.get(toLayer);
        
        if (source == null || target == null) return;
        
        int packetCount = Math.min(10, source.positions.size());
        Random random = new Random();
        
        for (int i = 0; i < packetCount; i++) {
            int srcIdx = random.nextInt(source.positions.size());
            int dstIdx = random.nextInt(target.positions.size());
            
            GradientPacket packet = new GradientPacket(
                source.positions.get(srcIdx),
                target.positions.get(dstIdx),
                intensity
            );
            gradientPackets.add(packet);
        }
    }
    
    /**
     * Renders and updates data flow packets.
     */
    private void renderDataPackets() {
        List<DataPacket> toRemove = new ArrayList<>();
        
        for (DataPacket packet : activePackets) {
            packet.update(Gdx.graphics.getDeltaTime());
            packet.render(modelBatch, environment);
            
            if (packet.isComplete()) {
                toRemove.add(packet);
            }
        }
        
        activePackets.removeAll(toRemove);
        for (DataPacket packet : toRemove) {
            packet.dispose();
        }
    }
    
    /**
     * Renders and updates gradient packets.
     */
    private void renderGradientPackets() {
        List<GradientPacket> toRemove = new ArrayList<>();
        
        for (GradientPacket packet : gradientPackets) {
            packet.update(Gdx.graphics.getDeltaTime());
            packet.render(modelBatch, environment);
            
            if (packet.isComplete()) {
                toRemove.add(packet);
            }
        }
        
        gradientPackets.removeAll(toRemove);
        for (GradientPacket packet : toRemove) {
            packet.dispose();
        }
    }
    
    // ==================== EVENT PROCESSING ====================
    
    private void processEvents() {
        VisualizationEvent event;
        while ((event = eventQueue.poll()) != null) {
            switch (event.type) {
                case FORWARD_PASS:
                    processForwardPass(event.data);
                    break;
                case BACKWARD_PASS:
                    processBackwardPass(event.data);
                    break;
                case METRICS_UPDATE:
                    processMetricsUpdate(event.data);
                    break;
                case LAYER_HIGHLIGHT:
                    processLayerHighlight(event.data);
                    break;
            }
        }
    }
    
    private void processForwardPass(Object[] data) {
        float[][] activations = (float[][]) data;
        
        // Animate through layers
        int layerIdx = 0;
        for (LayerVisualizer layer : layerVisualizers) {
            final int idx = layerIdx;
            layer.setActivation(activations[0], activations[1]); // Simplified
            
            // Spawn data packets to next layer
            if (layerIdx < layerVisualizers.size() - 1) {
                String currentName = layer.group.layerName;
                String nextName = layerVisualizers.get(layerIdx + 1).group.layerName;
                spawnDataPackets(currentName, nextName, 0.5f);
            }
            
            layerIdx++;
        }
    }
    
    private void processBackwardPass(Object[] data) {
        float[] gradients = (float[]) data[0];
        
        // Animate backward from output to input
        for (int i = layerVisualizers.size() - 1; i > 0; i--) {
            String currentName = layerVisualizers.get(i).group.layerName;
            String prevName = layerVisualizers.get(i - 1).group.layerName;
            spawnGradientPackets(currentName, prevName, 0.7f);
        }
    }
    
    private void processMetricsUpdate(Object[] data) {
        currentLoss = (Float) data[0];
        currentAccuracy = (Float) data[1];
        currentEpoch = (Integer) data[2];
        currentStep = (Integer) data[3];
        currentPhase = (String) data[4];
    }
    
    private void processLayerHighlight(Object[] data) {
        String layerName = (String) data[0];
        float intensity = (Float) data[1];
        
        for (LayerVisualizer layer : layerVisualizers) {
            if (layer.group.layerName.equals(layerName)) {
                layer.setHighlight(intensity);
                break;
            }
        }
    }
    
    // ==================== ENGINE STATE POLLING ====================
    
    private void pollEngineState() {
        // This would query the engine for current state
        // For now, we'll use placeholder logic
    }
    
    // ==================== OVERLAY RENDERING ====================
    
    private void renderOverlay() {
        // Render metrics as a simple overlay using 3D positions projected to 2D
        // For now, just log to console in debug mode
    }
    
    // ==================== ENUMS AND INNER CLASSES ====================
    
    private enum LayerType {
        CONV2D, BILSTM, DENSE, UNKNOWN
    }
    
    private enum EventType {
        FORWARD_PASS, BACKWARD_PASS, METRICS_UPDATE, LAYER_HIGHLIGHT
    }
    
    private static class VisualizationEvent {
        final EventType type;
        final Object data;
        
        VisualizationEvent(EventType type, Object data) {
            this.type = type;
            this.data = data;
        }
    }
    
    /**
     * Group of neurons representing a layer.
     */
    private static class NeuronGroup {
        String layerName;
        LayerType layerType;
        float zPosition;
        Color color;
        List<Vector3> positions = new ArrayList<>();
    }
    
    /**
     * Group of connections between two layers.
     */
    private static class ConnectionGroup implements Disposable {
        NeuronGroup source;
        NeuronGroup target;
        List<ModelInstance> lineModels = new ArrayList<>();
        
        ConnectionGroup(NeuronGroup source, NeuronGroup target) {
            this.source = source;
            this.target = target;
            buildConnections();
        }
        
        private void buildConnections() {
            // Sample connections for performance
            int maxConnections = 100;
            int step = Math.max(1, source.positions.size() * target.positions.size() / maxConnections);
            
            ModelBuilder builder = new ModelBuilder();
            builder.begin();
            
            for (int i = 0; i < source.positions.size(); i += step) {
                for (int j = 0; j < target.positions.size(); j += step) {
                    Vector3 start = source.positions.get(i);
                    Vector3 end = target.positions.get(j);
                    
                    builder.addLine(start, end, new Color(0.2f, 0.2f, 0.3f, 0.3f));
                }
            }
            
            Model model = builder.end();
            lineModels.add(new ModelInstance(model));
        }
        
        void render(ModelBatch batch, Environment env) {
            for (ModelInstance instance : lineModels) {
                batch.render(instance, env);
            }
        }
        
        public void dispose() {
            for (ModelInstance instance : lineModels) {
                instance.model.dispose();
            }
        }
    }
    
    /**
     * Data packet for forward pass animation.
     */
    private static class DataPacket implements Disposable {
        Vector3 start, end, current;
        float progress;
        float speed;
        Color color;
        float size = 0.15f;
        
        DataPacket(Vector3 start, Vector3 end, float intensity, Color color) {
            this.start = start;
            this.end = end;
            this.current = new Vector3(start);
            this.progress = 0;
            this.speed = 2f;
            this.color = new Color(color.r * intensity, color.g * intensity, color.b * intensity, 1f);
        }
        
        void update(float deltaTime) {
            progress += deltaTime * speed;
            if (progress > 1) progress = 1;
            
            current.lerp(start, end, progress);
        }
        
        void render(ModelBatch batch, Environment env) {
            if (progress >= 1) return;
            
            // Create a simple sphere for the packet
            ModelBuilder builder = new ModelBuilder();
            builder.begin();
            builder.addSphere(current.x, current.y, current.z, size, size, size, 8, 8);
            Model model = builder.end();
            
            ModelInstance instance = new ModelInstance(model);
            instance.material.set(ColorAttribute.createEmissive(color));
            batch.render(instance, env);
            
            model.dispose();
            instance.dispose();
        }
        
        boolean isComplete() {
            return progress >= 1;
        }
        
        public void dispose() {
            // Clean up if needed
        }
    }
    
    /**
     * Gradient packet for backpropagation animation.
     */
    private static class GradientPacket implements Disposable {
        Vector3 start, end, current;
        float progress;
        float speed;
        
        GradientPacket(Vector3 start, Vector3 end, float intensity) {
            this.start = start;
            this.end = end;
            this.current = new Vector3(start);
            this.progress = 0;
            this.speed = 1.5f; // Slightly slower than forward pass
        }
        
        void update(float deltaTime) {
            progress += deltaTime * speed;
            if (progress > 1) progress = 1;
            
            current.lerp(start, end, progress);
        }
        
        void render(ModelBatch batch, Environment env) {
            if (progress >= 1) return;
            
            ModelBuilder builder = new ModelBuilder();
            builder.begin();
            // Red tinted packet for gradients
            builder.addSphere(current.x, current.y, current.z, 0.12f, 0.12f, 0.12f, 8, 8);
            Model model = builder.end();
            
            ModelInstance instance = new ModelInstance(model);
            instance.material.set(ColorAttribute.createEmissive(1f, 0.2f, 0.2f, 1f));
            batch.render(instance, env);
            
            model.dispose();
            instance.dispose();
        }
        
        boolean isComplete() {
            return progress >= 1;
        }
        
        public void dispose() {
            // Clean up if needed
        }
    }
}
