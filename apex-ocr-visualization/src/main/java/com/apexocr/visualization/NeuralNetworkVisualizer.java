package com.apexocr.visualization;

import com.apexocr.training.monitoring.NetworkArchitecture;
import com.apexocr.training.monitoring.TrainingSnapshot;
import com.apexocr.training.monitoring.VisualizationService;
import com.badlogic.gdx.ApplicationAdapter;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.GL30;
import com.badlogic.gdx.graphics.g3d.Environment;
import com.badlogic.gdx.graphics.g3d.ModelBatch;
import com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute;
import com.badlogic.gdx.graphics.g3d.environment.DirectionalLight;
import com.badlogic.gdx.graphics.g3d.utils.ModelBuilder;
import com.badlogic.gdx.input.GestureDetector;
import com.badlogic.gdx.math.Vector3;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * NeuralNetworkVisualizer - Main entry point for the 3D visualization system.
 * 
 * Features:
 * - 3D representation of the neural network architecture
 * - Real-time training metrics display
 * - Animated forward/backward pass visualization
 * - Interactive camera controls (orbit, zoom, pan)
 * - Opens automatically when training is active
 *
 * @author ApexOCR Team
 */
public class NeuralNetworkVisualizer extends ApplicationAdapter {
    
    // ==================== STATIC ACCESS FOR EXTERNAL CONTROL ====================
    
    private static NeuralNetworkVisualizer instance;
    private static AtomicBoolean isRunning = new AtomicBoolean(false);
    private static String[] startupArgs = new String[0];
    
    /**
     * Gets the singleton instance of the visualizer.
     */
    public static NeuralNetworkVisualizer getInstance() {
        return instance;
    }
    
    /**
     * Check if visualizer is currently running.
     */
    public static boolean isRunning() {
        return isRunning.get();
    }
    
    /**
     * Opens the 3D visualization window with default architecture.
     */
    public static void open() {
        open(null, null);
    }
    
    /**
     * Opens the 3D visualization window with a custom architecture.
     * @param architecture Network architecture description
     */
    public static void open(NetworkArchitecture architecture) {
        open(architecture, null);
    }
    
    /**
     * Opens the 3D visualization window connected to training service.
     * This is the main entry point when starting from training code.
     * @param architecture Network architecture description
     * @param args Command line arguments
     */
    public static void open(NetworkArchitecture architecture, String[] args) {
        if (isRunning.get()) {
            System.out.println("[NeuralNetworkVisualizer] Already running");
            return;
        }
        
        if (args != null) {
            startupArgs = args;
        }
        
        Lwjgl3ApplicationConfiguration config = new Lwjgl3ApplicationConfiguration();
        config.setTitle("Neural Network Visualizer - ApexOCR");
        config.setWindowSizeLimits(1280, 720, 1920, 1080);
        config.setResizable(true);
        config.useVsync(true);
        config.setForegroundFPS(60);
        
        instance = new NeuralNetworkVisualizer(architecture);
        isRunning.set(true);
        
        new Lwjgl3Application(instance, config);
    }
    
    /**
     * Opens the visualizer connected to the training monitoring service.
     * This is the recommended method for training visualization.
     */
    public static void openForTraining(NetworkArchitecture architecture) {
        // Register with the visualization service
        VisualizationService service = VisualizationService.getInstance();
        service.setVisualizerConnected(true);
        
        open(architecture, new String[]{"--training"});
    }
    
    /**
     * Closes the visualization window.
     */
    public static void close() {
        if (instance != null) {
            instance.dispose();
            instance = null;
            isRunning.set(false);
        }
    }
    
    // ==================== INSTANCE VARIABLES ====================
    
    // 3D Rendering
    private ModelBatch modelBatch;
    private NetworkRenderer networkRenderer;
    private HUDOverlay hudOverlay;
    
    // State
    private NetworkArchitecture networkArchitecture;
    private boolean isTrainingMode = false;
    private float timeElapsed = 0f;
    
    // Camera controls
    private float cameraDistance = 25f;
    private float cameraAngleY = 0f;
    private float cameraAngleX = 30f;
    private boolean isDragging = false;
    private int lastMouseX, lastMouseY;
    
    // Colors
    private static final Color BACKGROUND_COLOR = new Color(0.05f, 0.06f, 0.08f, 1f);
    
    // ==================== CONSTRUCTOR ====================
    
    public NeuralNetworkVisualizer(NetworkArchitecture architecture) {
        this.networkArchitecture = architecture;
        this.isTrainingMode = false;
    }
    
    // ==================== LIFECYCLE METHODS ====================
    
    @Override
    public void create() {
        System.out.println("[NeuralNetworkVisualizer] Initializing 3D visualization...");
        
        // Initialize 3D rendering
        initializeRendering();
        
        // Initialize network architecture
        if (networkArchitecture != null) {
            networkRenderer.setNetworkArchitecture(networkArchitecture);
            System.out.println("[NeuralNetworkVisualizer] Network architecture loaded: " + 
                networkArchitecture.name + " with " + networkArchitecture.getLayerCount() + " layers");
        } else {
            // Create default architecture for demo
            createDefaultArchitecture();
        }
        
        // Initialize HUD
        hudOverlay = new HUDOverlay();
        hudOverlay.resize(Gdx.graphics.getWidth(), Gdx.graphics.getHeight());
        
        // Set up input handling
        setupInput();
        
        // Register with visualization service
        VisualizationService.getInstance().setVisualizerConnected(true);
        
        System.out.println("[NeuralNetworkVisualizer] 3D visualization initialized successfully");
    }
    
    @Override
    public void render() {
        float delta = Gdx.graphics.getDeltaTime();
        timeElapsed += delta;
        
        // Clear screen
        Gdx.gl.glClearColor(BACKGROUND_COLOR.r, BACKGROUND_COLOR.g, BACKGROUND_COLOR.b, 1f);
        Gdx.gl.glClear(GL30.GL_COLOR_BUFFER_BIT | GL30.GL_DEPTH_BUFFER_BIT);
        Gdx.gl.glEnable(GL30.GL_DEPTH_TEST);
        Gdx.gl.glEnable(GL30.GL_BLEND);
        Gdx.gl.glBlendFunc(GL30.GL_SRC_ALPHA, GL30.GL_ONE_MINUS_SRC_ALPHA);
        
        // Update from visualization service (training mode)
        if (isTrainingMode) {
            updateFromTrainingService();
        }
        
        // Render 3D scene
        networkRenderer.render(delta);
        
        // Render 2D HUD overlay
        Gdx.gl.glDisable(GL30.GL_DEPTH_TEST);
        hudOverlay.render(VisualizationService.getInstance());
        
        // Handle camera input
        handleCameraInput();
        
        // Update window title with metrics
        updateWindowTitle();
    }
    
    @Override
    public void resize(int width, int height) {
        if (networkRenderer != null && networkRenderer.getCamera() != null) {
            networkRenderer.getCamera().viewportWidth = width;
            networkRenderer.getCamera().viewportHeight = height;
            networkRenderer.getCamera().update();
        }
        
        if (hudOverlay != null) {
            hudOverlay.resize(width, height);
        }
    }
    
    @Override
    public void dispose() {
        System.out.println("[NeuralNetworkVisualizer] Disposing resources...");
        
        // Unregister from visualization service
        VisualizationService.getInstance().setVisualizerConnected(false);
        
        if (networkRenderer != null) {
            networkRenderer.dispose();
        }
        
        if (hudOverlay != null) {
            hudOverlay.dispose();
        }
        
        if (modelBatch != null) {
            modelBatch.dispose();
        }
        
        isRunning.set(false);
        System.out.println("[NeuralNetworkVisualizer] Resources disposed");
    }
    
    // ==================== INITIALIZATION ====================
    
    private void initializeRendering() {
        // Create model batch for 3D rendering
        modelBatch = new ModelBatch();
        
        // Create network renderer
        networkRenderer = new NetworkRenderer();
    }
    
    private void createDefaultArchitecture() {
        // Create a default architecture for demonstration
        NetworkArchitecture.Builder builder = NetworkArchitecture.builder();
        builder.setName("Demo CRNN");
        
        // Input layer
        builder.addLayer(
            NetworkArchitecture.LayerInfo.builder()
                .setName("Input")
                .setType(com.apexocr.training.monitoring.LayerSnapshot.LayerType.INPUT)
                .setDimensions(1, 64, 32, 128)
                .setPosition(-3f, 0f, 0f)
                .build()
        );
        
        // Conv2D layers
        builder.addLayer(
            NetworkArchitecture.LayerInfo.builder()
                .setName("Conv2D_1")
                .setType(com.apexocr.training.monitoring.LayerSnapshot.LayerType.CONV2D)
                .setDimensions(64, 32, 16, 64)
                .setKernelInfo(3, 1)
                .setPosition(-2f, 0f, 0f)
                .build()
        );
        
        builder.addLayer(
            NetworkArchitecture.LayerInfo.builder()
                .setName("Conv2D_2")
                .setType(com.apexocr.training.monitoring.LayerSnapshot.LayerType.CONV2D)
                .setDimensions(32, 64, 8, 32)
                .setKernelInfo(3, 1)
                .setPosition(-1f, 0f, 0f)
                .build()
        );
        
        // BiLSTM layers
        builder.addLayer(
            NetworkArchitecture.LayerInfo.builder()
                .setName("BiLSTM_1")
                .setType(com.apexocr.training.monitoring.LayerSnapshot.LayerType.BILSTM)
                .setDimensions(64, 256, 1, 32)
                .setPosition(0f, 0f, 0f)
                .build()
        );
        
        builder.addLayer(
            NetworkArchitecture.LayerInfo.builder()
                .setName("BiLSTM_2")
                .setType(com.apexocr.training.monitoring.LayerSnapshot.LayerType.BILSTM)
                .setDimensions(256, 256, 1, 32)
                .setPosition(1f, 0f, 0f)
                .build()
        );
        
        // Dense output layer
        builder.addLayer(
            NetworkArchitecture.LayerInfo.builder()
                .setName("Output")
                .setType(com.apexocr.training.monitoring.LayerSnapshot.LayerType.OUTPUT)
                .setDimensions(256, 37, 1, 1)
                .setPosition(2f, 0f, 0f)
                .build()
        );
        
        builder.setInputSize(4096);
        builder.setOutputSize(37);
        builder.setTotalParameters(245000);
        
        networkArchitecture = builder.build();
        networkRenderer.setNetworkArchitecture(networkArchitecture);
        
        System.out.println("[NeuralNetworkVisualizer] Created default architecture with " + 
            networkArchitecture.getLayerCount() + " layers");
    }
    
    private void setupInput() {
        // Set up mouse input for camera control
        Gdx.input.setInputProcessor(new GestureDetector(new GestureDetector.GestureAdapter() {
            @Override
            public boolean pan(float x, float y, float deltaX, float deltaY) {
                cameraAngleY += deltaX * 0.5f;
                cameraAngleX += deltaY * 0.5f;
                cameraAngleX = Math.max(-89f, Math.min(89f, cameraAngleX));
                networkRenderer.setCameraAngles(cameraAngleY, cameraAngleX);
                return true;
            }
            
            @Override
            public boolean zoom(float initialDistance, float distance) {
                float ratio = initialDistance / distance;
                cameraDistance *= ratio;
                cameraDistance = Math.max(5f, Math.min(100f, cameraDistance));
                networkRenderer.setCameraDistance(cameraDistance);
                return true;
            }
            
            @Override
            public boolean tap(float x, float y, int count, int button) {
                if (count == 2) {
                    // Double tap to reset camera
                    networkRenderer.resetCamera();
                    cameraDistance = 25f;
                    cameraAngleY = 0f;
                    cameraAngleX = 30f;
                }
                return true;
            }
        }));
    }
    
    private void handleCameraInput() {
        // Keyboard controls
        if (Gdx.input.isKeyPressed(com.badlogic.gdx.Input.Keys.SPACE)) {
            // Toggle pause
            VisualizationService.getInstance().togglePause();
        }
        
        if (Gdx.input.isKeyPressed(com.badlogic.gdx.Input.Keys.R)) {
            // Reset camera
            networkRenderer.resetCamera();
            cameraDistance = 25f;
            cameraAngleY = 0f;
            cameraAngleX = 30f;
        }
        
        if (Gdx.input.isKeyPressed(com.badlogic.gdx.Input.Keys.ESCAPE)) {
            // Exit
            close();
        }
    }
    
    // ==================== TRAINING INTEGRATION ====================
    
    private void updateFromTrainingService() {
        VisualizationService service = VisualizationService.getInstance();
        
        // Get latest snapshot from training
        service.getLatestSnapshot().ifPresent(snapshot -> {
            networkRenderer.updateFromSnapshot(snapshot);
        });
    }
    
    private void updateWindowTitle() {
        VisualizationService service = VisualizationService.getInstance();
        float loss = service.getCurrentLoss();
        float accuracy = service.getCurrentAccuracy() * 100f;
        int epoch = service.getCurrentEpoch();
        int totalEpochs = service.getTotalEpochs();
        
        String title = String.format("ApexOCR Training - Epoch %d/%d | Loss: %.6f | Accuracy: %.2f%%",
            epoch + 1, totalEpochs, loss, accuracy);
        
        Gdx.graphics.setTitle(title);
    }
    
    // ==================== STATIC HELPER METHODS ====================
    
    /**
     * Utility method to create a simple line for debugging.
     */
    private static ModelInstance createLine(Vector3 start, Vector3 end, Color color) {
        ModelBuilder builder = new ModelBuilder();
        builder.begin();
        builder.addLine(start, end, color);
        return new ModelInstance(builder.end());
    }
}
