package com.apexocr.visualization;

import com.apexocr.core.monitoring.NetworkArchitecture;
import com.apexocr.core.monitoring.TrainingSnapshot;
import com.apexocr.core.monitoring.VisualizationService;
import com.badlogic.gdx.ApplicationAdapter;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application;
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.GL30;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer.ShapeType;
import com.badlogic.gdx.math.Matrix4;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * NeuralNetworkVisualizer - 2D-based visualization for neural network training.
 * 
 * Features:
 * - 2D representation of the neural network architecture
 * - Real-time training metrics display
 * - Animated activation flow visualization
 * - Layer statistics visualization
 *
 * @author ApexOCR Team
 */
public class NeuralNetworkVisualizer extends ApplicationAdapter {
    
    // ==================== STATIC ACCESS FOR EXTERNAL CONTROL ====================
    
    private static NeuralNetworkVisualizer instance;
    private static AtomicBoolean isRunning = new AtomicBoolean(false);
    private static String[] startupArgs = new String[0];
    
    public static NeuralNetworkVisualizer getInstance() {
        return instance;
    }
    
    public static boolean isRunning() {
        return isRunning.get();
    }
    
    public static void open() {
        open(null, null);
    }
    
    public static void open(NetworkArchitecture architecture) {
        open(architecture, null);
    }
    
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
    
    public static void openForTraining(NetworkArchitecture architecture) {
        VisualizationService service = VisualizationService.getInstance();
        service.setVisualizerConnected(true);
        open(architecture, new String[]{"--training"});
    }
    
    public static void close() {
        if (instance != null) {
            instance.dispose();
            instance = null;
            isRunning.set(false);
        }
    }
    
    // ==================== INSTANCE VARIABLES ====================
    
    private SpriteBatch spriteBatch;
    private ShapeRenderer shapeRenderer;
    private BitmapFont font;
    
    private NetworkArchitecture networkArchitecture;
    private boolean isTrainingMode = false;
    private float timeElapsed = 0f;
    
    // Layout constants
    private static final int LAYER_WIDTH = 80;
    private static final int LAYER_HEIGHT = 200;
    private static final int LAYER_SPACING = 120;
    private static final int START_X = 100;
    private static final int CENTER_Y = 350;
    
    // Colors
    private static final Color BACKGROUND_COLOR = new Color(0.1f, 0.1f, 0.15f, 1f);
    private static final Color LAYER_COLOR = new Color(0.2f, 0.4f, 0.8f, 0.8f);
    private static final Color LAYER_ACTIVE_COLOR = new Color(0.2f, 0.8f, 0.4f, 0.9f);
    private static final Color CONNECTION_COLOR = new Color(0.3f, 0.3f, 0.4f, 0.5f);
    private static final Color TEXT_COLOR = Color.WHITE;
    private static final Color METRIC_COLOR = new Color(0.9f, 0.9f, 0.9f, 1f);
    
    // ==================== CONSTRUCTOR ====================
    
    public NeuralNetworkVisualizer(NetworkArchitecture architecture) {
        this.networkArchitecture = architecture;
        this.isTrainingMode = false;
    }
    
    // ==================== LIFECYCLE METHODS ====================
    
    @Override
    public void create() {
        System.out.println("[NeuralNetworkVisualizer] Initializing visualization...");
        
        // Initialize rendering
        spriteBatch = new SpriteBatch();
        shapeRenderer = new ShapeRenderer();
        font = new BitmapFont();
        font.setColor(TEXT_COLOR);
        
        // Initialize network architecture
        if (networkArchitecture != null) {
            System.out.println("[NeuralNetworkVisualizer] Network architecture loaded: " + 
                networkArchitecture.name + " with " + networkArchitecture.getLayerCount() + " layers");
        } else {
            createDefaultArchitecture();
        }
        
        // Register with visualization service
        VisualizationService.getInstance().setVisualizerConnected(true);
        
        System.out.println("[NeuralNetworkVisualizer] Visualization initialized successfully");
    }
    
    @Override
    public void render() {
        float delta = Gdx.graphics.getDeltaTime();
        timeElapsed += delta;
        
        // Clear screen
        Gdx.gl.glClearColor(BACKGROUND_COLOR.r, BACKGROUND_COLOR.g, BACKGROUND_COLOR.b, 1f);
        Gdx.gl.glClear(GL30.GL_COLOR_BUFFER_BIT);
        
        // Update from visualization service (training mode)
        if (isTrainingMode) {
            updateFromTrainingService();
        }
        
        // Render
        renderNetwork();
        renderHUD();
        
        // Handle input
        handleInput();
    }
    
    @Override
    public void resize(int width, int height) {
        spriteBatch.getProjectionMatrix().setToOrtho2D(0, 0, width, height);
    }
    
    @Override
    public void dispose() {
        System.out.println("[NeuralNetworkVisualizer] Disposing resources...");
        
        VisualizationService.getInstance().setVisualizerConnected(false);
        
        if (spriteBatch != null) spriteBatch.dispose();
        if (shapeRenderer != null) shapeRenderer.dispose();
        if (font != null) font.dispose();
        
        isRunning.set(false);
        System.out.println("[NeuralNetworkVisualizer] Resources disposed");
    }
    
    // ==================== RENDERING ====================
    
    private void renderNetwork() {
        if (networkArchitecture == null || networkArchitecture.layers == null) {
            return;
        }
        
        int numLayers = networkArchitecture.layers.size();
        float[] layerActivations = getCurrentLayerActivations();
        
        for (int i = 0; i < numLayers; i++) {
            NetworkArchitecture.LayerInfo layer = networkArchitecture.layers.get(i);
            int x = START_X + i * LAYER_SPACING;
            int y = CENTER_Y - LAYER_HEIGHT / 2;
            
            // Determine layer color based on activation
            Color layerColor = LAYER_COLOR;
            if (i < layerActivations.length && layerActivations[i] > 0) {
                float intensity = Math.min(1f, layerActivations[i]);
                layerColor = new Color(
                    LAYER_COLOR.r * (1 - intensity) + LAYER_ACTIVE_COLOR.r * intensity,
                    LAYER_COLOR.g * (1 - intensity) + LAYER_ACTIVE_COLOR.g * intensity,
                    LAYER_COLOR.b * (1 - intensity) + LAYER_ACTIVE_COLOR.b * intensity,
                    0.9f
                );
            }
            
            // Draw layer box
            shapeRenderer.begin(ShapeType.Filled);
            shapeRenderer.setColor(layerColor);
            shapeRenderer.rect(x, y, LAYER_WIDTH, LAYER_HEIGHT);
            shapeRenderer.end();
            
            // Draw layer border
            shapeRenderer.begin(ShapeType.Line);
            shapeRenderer.setColor(Color.WHITE);
            shapeRenderer.rect(x, y, LAYER_WIDTH, LAYER_HEIGHT);
            shapeRenderer.end();
            
            // Draw layer name
            spriteBatch.begin();
            font.draw(spriteBatch, layer.name, x, y + LAYER_HEIGHT + 20);
            font.draw(spriteBatch, getLayerTypeShort(layer.type), x + LAYER_WIDTH / 2 - 20, y - 10);
            spriteBatch.end();
            
            // Draw connections to next layer
            if (i < numLayers - 1) {
                int nextX = START_X + (i + 1) * LAYER_SPACING;
                shapeRenderer.begin(ShapeType.Line);
                shapeRenderer.setColor(CONNECTION_COLOR);
                shapeRenderer.line(x + LAYER_WIDTH, y + LAYER_HEIGHT / 2, nextX, CENTER_Y);
                shapeRenderer.end();
            }
        }
        
        // Draw input label
        spriteBatch.begin();
        font.draw(spriteBatch, "INPUT", START_X, CENTER_Y + LAYER_HEIGHT / 2 + 40);
        font.draw(spriteBatch, "OUTPUT", START_X + (numLayers - 1) * LAYER_SPACING, CENTER_Y + LAYER_HEIGHT / 2 + 40);
        spriteBatch.end();
    }
    
    private void renderHUD() {
        VisualizationService service = VisualizationService.getInstance();
        
        // Get current metrics
        float loss = service.getCurrentLoss();
        float accuracy = service.getCurrentAccuracy() * 100f;
        int epoch = service.getCurrentEpoch();
        int totalEpochs = service.getTotalEpochs();
        int batch = service.getCurrentBatch();
        int totalBatches = service.getTotalBatches();
        float lr = service.getCurrentLearningRate();
        
        // Draw HUD background
        shapeRenderer.begin(ShapeType.Filled);
        shapeRenderer.setColor(new Color(0f, 0f, 0f, 0.7f));
        shapeRenderer.rect(10, 10, 350, 200);
        shapeRenderer.end();
        
        // Draw HUD text
        spriteBatch.begin();
        font.draw(spriteBatch, "=== TRAINING METRICS ===", 25, 190);
        font.draw(spriteBatch, String.format("Epoch: %d / %d", epoch + 1, totalEpochs), 25, 165);
        font.draw(spriteBatch, String.format("Batch: %d / %d", batch, totalBatches), 25, 145);
        font.draw(spriteBatch, String.format("Loss: %.6f", loss), 25, 120);
        font.draw(spriteBatch, String.format("Accuracy: %.2f%%", accuracy), 25, 95);
        font.draw(spriteBatch, String.format("Learning Rate: %.6f", lr), 25, 70);
        
        // Draw status
        String status = service.isPaused() ? "PAUSED" : "RUNNING";
        Color statusColor = service.isPaused() ? Color.YELLOW : Color.GREEN;
        font.setColor(statusColor);
        font.draw(spriteBatch, "Status: " + status, 25, 40);
        font.setColor(TEXT_COLOR);
        
        // Draw network info
        font.draw(spriteBatch, "=== NETWORK INFO ===", 200, 190);
        String networkName = networkArchitecture != null ? networkArchitecture.name : "Unknown";
        font.draw(spriteBatch, "Name: " + networkName, 200, 165);
        int layerCount = networkArchitecture != null ? networkArchitecture.getLayerCount() : 0;
        font.draw(spriteBatch, "Layers: " + layerCount, 200, 145);
        int params = networkArchitecture != null ? networkArchitecture.totalParameters : 0;
        font.draw(spriteBatch, "Params: " + params, 200, 125);
        font.draw(spriteBatch, "Controls: SPACE=Pause R=Reset ESC=Exit", 200, 40);
        
        spriteBatch.end();
    }
    
    private void handleInput() {
        if (Gdx.input.isKeyPressed(com.badlogic.gdx.Input.Keys.SPACE)) {
            VisualizationService.getInstance().togglePause();
        }
        
        if (Gdx.input.isKeyPressed(com.badlogic.gdx.Input.Keys.R)) {
            // Reset view - could reset animations here
        }
        
        if (Gdx.input.isKeyPressed(com.badlogic.gdx.Input.Keys.ESCAPE)) {
            close();
        }
    }
    
    // ==================== TRAINING INTEGRATION ====================
    
    private void updateFromTrainingService() {
        // Check if we have a snapshot
        VisualizationService service = VisualizationService.getInstance();
        service.getLatestSnapshot().ifPresent(snapshot -> {
            // Update window title
            String title = String.format("ApexOCR Training - Epoch %d/%d | Loss: %.6f | Accuracy: %.2f%%",
                snapshot.epoch + 1, snapshot.totalEpochs, snapshot.currentLoss, snapshot.currentAccuracy * 100f);
            Gdx.graphics.setTitle(title);
        });
    }
    
    private float[] getCurrentLayerActivations() {
        VisualizationService service = VisualizationService.getInstance();
        float[] activations = new float[networkArchitecture != null ? networkArchitecture.getLayerCount() : 5];
        
        service.getLatestSnapshot().ifPresent(snapshot -> {
            if (snapshot.layerSnapshots != null) {
                int idx = 0;
                for (var entry : snapshot.layerSnapshots.entrySet()) {
                    if (idx < activations.length) {
                        // Use activation mean as intensity
                        activations[idx] = Math.abs(entry.getValue().activationMean);
                        idx++;
                    }
                }
            }
        });
        
        return activations;
    }
    
    // ==================== HELPER METHODS ====================
    
    private void createDefaultArchitecture() {
        NetworkArchitecture.Builder builder = NetworkArchitecture.builder();
        builder.setName("Demo CRNN");
        
        builder.addLayer(
            new NetworkArchitecture.LayerInfo.Builder()
                .setName("Input")
                .setType(com.apexocr.core.monitoring.LayerSnapshot.LayerType.INPUT)
                .setDimensions(1, 64, 32, 128)
                .build()
        );
        
        builder.addLayer(
            new NetworkArchitecture.LayerInfo.Builder()
                .setName("Conv2D_1")
                .setType(com.apexocr.core.monitoring.LayerSnapshot.LayerType.CONV2D)
                .setDimensions(64, 32, 16, 64)
                .setKernelInfo(3, 1)
                .build()
        );
        
        builder.addLayer(
            new NetworkArchitecture.LayerInfo.Builder()
                .setName("BiLSTM")
                .setType(com.apexocr.core.monitoring.LayerSnapshot.LayerType.BILSTM)
                .setDimensions(32, 256, 1, 32)
                .build()
        );
        
        builder.addLayer(
            new NetworkArchitecture.LayerInfo.Builder()
                .setName("Dense")
                .setType(com.apexocr.core.monitoring.LayerSnapshot.LayerType.DENSE)
                .setDimensions(256, 37, 1, 1)
                .build()
        );
        
        builder.addLayer(
            new NetworkArchitecture.LayerInfo.Builder()
                .setName("Output")
                .setType(com.apexocr.core.monitoring.LayerSnapshot.LayerType.OUTPUT)
                .setDimensions(37, 37, 1, 1)
                .build()
        );
        
        builder.setInputSize(4096);
        builder.setOutputSize(37);
        builder.setTotalParameters(245000);
        
        networkArchitecture = builder.build();
        
        System.out.println("[NeuralNetworkVisualizer] Created default architecture with " + 
            networkArchitecture.getLayerCount() + " layers");
    }
    
    private String getLayerTypeShort(com.apexocr.core.monitoring.LayerSnapshot.LayerType type) {
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
