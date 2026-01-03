package com.apexocr.visualization;

import com.apexocr.core.monitoring.NetworkArchitecture;
import com.apexocr.core.monitoring.TrainingSnapshot;
import com.apexocr.core.monitoring.LayerSnapshot;
import com.apexocr.core.monitoring.VisualizationService;
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
import com.badlogic.gdx.graphics.g3d.utils.CameraInputController;
import com.badlogic.gdx.math.Vector3;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * NeuralNetworkVisualizer - 3D-based visualization for neural network training.
 * 
 * Features:
 * - 3D representation of the neural network architecture
 * - Real-time training metrics display
 * - Animated activation flow visualization
 * - Layer statistics visualization
 * - Interactive camera controls
 *
 * @author ApexOCR Team
 */
public class NeuralNetworkVisualizer extends ApplicationAdapter {
    
    // ==================== STATIC ACCESS FOR EXTERNAL CONTROL ====================
    
    private static NeuralNetworkVisualizer instance;
    private static AtomicBoolean isRunning = new AtomicBoolean(false);
    private static AtomicReference<String[]> startupArgs = new AtomicReference<>(new String[0]);
    
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
            startupArgs.set(args);
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
    
    private NetworkArchitecture networkArchitecture;
    private NetworkRenderer networkRenderer;
    private HUDOverlay hudOverlay;
    
    private boolean isTrainingMode = false;
    private float timeElapsed = 0f;
    
    // ==================== CONSTRUCTOR ====================
    
    public NeuralNetworkVisualizer(NetworkArchitecture architecture) {
        this.networkArchitecture = architecture;
    }
    
    // ==================== LIFECYCLE METHODS ====================
    
    @Override
    public void create() {
        System.out.println("[NeuralNetworkVisualizer] Initializing 3D visualization...");
        
        // Initialize network renderer
        networkRenderer = new NetworkRenderer();
        if (networkArchitecture != null) {
            networkRenderer.setNetworkArchitecture(networkArchitecture);
            System.out.println("[NeuralNetworkVisualizer] Network architecture loaded: " + 
                networkArchitecture.name + " with " + networkArchitecture.getLayerCount() + " layers");
        } else {
            createDefaultArchitecture();
        }
        
        // Initialize HUD overlay
        hudOverlay = new HUDOverlay();
        
        // Register with visualization service
        VisualizationService.getInstance().setVisualizerConnected(true);
        
        System.out.println("[NeuralNetworkVisualizer] 3D visualization initialized successfully");
    }
    
    @Override
    public void render() {
        float delta = Gdx.graphics.getDeltaTime();
        timeElapsed += delta;
        
        // Clear screen
        Gdx.gl.glClearColor(0.1f, 0.1f, 0.15f, 1f);
        Gdx.gl.glClear(GL30.GL_COLOR_BUFFER_BIT | GL30.GL_DEPTH_BUFFER_BIT);
        
        // Update from visualization service (training mode)
        if (isTrainingMode) {
            updateFromTrainingService();
        }
        
        // Render 3D scene
        if (networkRenderer != null) {
            networkRenderer.render(delta);
        }
        
        // Render HUD overlay
        if (hudOverlay != null) {
            hudOverlay.render(VisualizationService.getInstance());
        }
    }
    
    @Override
    public void resize(int width, int height) {
        if (networkRenderer != null) {
            networkRenderer.resize(width, height);
        }
        if (hudOverlay != null) {
            hudOverlay.resize(width, height);
        }
    }
    
    @Override
    public void dispose() {
        System.out.println("[NeuralNetworkVisualizer] Disposing resources...");
        
        VisualizationService.getInstance().setVisualizerConnected(false);
        
        if (networkRenderer != null) {
            networkRenderer.dispose();
        }
        if (hudOverlay != null) {
            hudOverlay.dispose();
        }
        
        isRunning.set(false);
        System.out.println("[NeuralNetworkVisualizer] Resources disposed");
    }
    
    // ==================== TRAINING INTEGRATION ====================
    
    private void updateFromTrainingService() {
        VisualizationService service = VisualizationService.getInstance();
        service.getLatestSnapshot().ifPresent(snapshot -> {
            // Update window title
            String title = String.format("ApexOCR Training - Epoch %d/%d | Loss: %.6f | Accuracy: %.2f%%",
                snapshot.epoch + 1, snapshot.totalEpochs, snapshot.currentLoss, snapshot.currentAccuracy * 100f);
            Gdx.graphics.setTitle(title);
            
            // Update 3D visualization
            if (networkRenderer != null) {
                networkRenderer.updateFromSnapshot(snapshot);
            }
        });
    }
    
    private void createDefaultArchitecture() {
        NetworkArchitecture.Builder builder = NetworkArchitecture.builder();
        builder.setName("Demo CRNN");
        
        builder.addLayer(
            new NetworkArchitecture.LayerInfo.Builder()
                .setName("Input")
                .setType(LayerSnapshot.LayerType.INPUT)
                .setDimensions(1, 64, 32, 128)
                .setPosition(0, 0, 0)
                .build()
        );
        
        builder.addLayer(
            new NetworkArchitecture.LayerInfo.Builder()
                .setName("Conv2D_1")
                .setType(LayerSnapshot.LayerType.CONV2D)
                .setDimensions(64, 32, 16, 64)
                .setKernelInfo(3, 1)
                .setPosition(10, 0, 0)
                .build()
        );
        
        builder.addLayer(
            new NetworkArchitecture.LayerInfo.Builder()
                .setName("BiLSTM")
                .setType(LayerSnapshot.LayerType.BILSTM)
                .setDimensions(32, 256, 1, 32)
                .setPosition(20, 0, 0)
                .build()
        );
        
        builder.addLayer(
            new NetworkArchitecture.LayerInfo.Builder()
                .setName("Dense")
                .setType(LayerSnapshot.LayerType.DENSE)
                .setDimensions(256, 37, 1, 1)
                .setPosition(30, 0, 0)
                .build()
        );
        
        builder.addLayer(
            new NetworkArchitecture.LayerInfo.Builder()
                .setName("Output")
                .setType(LayerSnapshot.LayerType.OUTPUT)
                .setDimensions(37, 37, 1, 1)
                .setPosition(40, 0, 0)
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
}
