package com.apexocr.visualization;

import com.apexocr.core.monitoring.LayerSnapshot;
import com.apexocr.core.monitoring.VisualizationService;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer.ShapeType;
import com.badlogic.gdx.math.Rectangle;

import java.util.ArrayList;
import java.util.List;

/**
 * 2D HUD overlay for displaying training metrics and network information.
 * Renders on top of the 3D visualization.
 */
public class HUDOverlay {
    
    private SpriteBatch spriteBatch;
    private ShapeRenderer shapeRenderer;
    private BitmapFont titleFont;
    private BitmapFont metricsFont;
    private BitmapFont smallFont;
    
    // Colors
    private static final Color BACKGROUND_COLOR = new Color(0f, 0f, 0f, 0.7f);
    private static final Color BORDER_COLOR = new Color(0.3f, 0.5f, 0.8f, 1f);
    private static final Color TEXT_COLOR = Color.WHITE;
    private static final Color VALUE_COLOR = new Color(0.4f, 0.9f, 0.4f, 1f);
    private static final Color WARNING_COLOR = new Color(1.0f, 0.8f, 0.2f, 1f);
    
    // Layout
    private int screenWidth;
    private int screenHeight;
    private float padding = 10f;
    
    // UI Elements
    private Rectangle metricsPanel;
    private Rectangle networkPanel;
    private Rectangle controlsPanel;
    private Rectangle lossGraphArea;
    
    // Loss history for graph
    private List<Float> lossHistory;
    private static final int MAX_LOSS_HISTORY = 100;
    
    // Current loss value
    private float currentLoss;
    
    public HUDOverlay() {
        initializeRendering();
        initializeLayout();
        initializeData();
    }
    
    private void initializeRendering() {
        spriteBatch = new SpriteBatch();
        shapeRenderer = new ShapeRenderer();
        
        // Use default bitmap fonts (no FreeType dependency)
        titleFont = new BitmapFont();
        metricsFont = new BitmapFont();
        smallFont = new BitmapFont();
        
        titleFont.setColor(TEXT_COLOR);
        metricsFont.setColor(TEXT_COLOR);
        smallFont.setColor(TEXT_COLOR);
    }
    
    private void initializeLayout() {
        screenWidth = Gdx.graphics.getWidth();
        screenHeight = Gdx.graphics.getHeight();
        
        // Define panel positions
        metricsPanel = new Rectangle(padding, padding, 280, 200);
        networkPanel = new Rectangle(padding, screenHeight - 220, 280, 200);
        controlsPanel = new Rectangle(screenWidth - 220, padding, 200, 120);
        lossGraphArea = new Rectangle(screenWidth - 320, screenHeight - 170, 300, 150);
    }
    
    private void initializeData() {
        lossHistory = new ArrayList<>();
        currentLoss = 0f;
    }
    
    public void resize(int width, int height) {
        screenWidth = width;
        screenHeight = height;
        initializeLayout();
    }
    
    public void render(VisualizationService service) {
        // Update current loss
        currentLoss = service.getCurrentLoss();
        
        // Update loss history
        if (lossHistory.size() >= MAX_LOSS_HISTORY) {
            lossHistory.remove(0);
        }
        lossHistory.add(currentLoss);
        
        // Begin rendering
        spriteBatch.begin();
        shapeRenderer.begin(ShapeType.Filled);
        
        // Draw all panels
        drawMetricsPanel(service, metricsPanel);
        drawNetworkPanel(service, networkPanel);
        drawControlsPanel(controlsPanel);
        drawLossGraph(lossGraphArea);
        
        // End rendering
        shapeRenderer.end();
        spriteBatch.end();
    }
    
    private void drawMetricsPanel(VisualizationService service, Rectangle panel) {
        // Draw panel background
        shapeRenderer.setColor(BACKGROUND_COLOR);
        shapeRenderer.rect(panel.x, panel.y, panel.width, panel.height);
        
        // Draw panel border
        shapeRenderer.end();
        shapeRenderer.begin(ShapeType.Line);
        shapeRenderer.setColor(BORDER_COLOR);
        shapeRenderer.rect(panel.x, panel.y, panel.width, panel.height);
        shapeRenderer.end();
        shapeRenderer.begin(ShapeType.Filled);
        
        // Draw title
        spriteBatch.enableBlending();
        titleFont.draw(spriteBatch, "=== TRAINING METRICS ===", 
            panel.x + padding, panel.y + panel.height - padding * 1.5f);
        
        // Draw metrics
        float y = panel.y + panel.height - padding * 3.5f;
        float lineHeight = 22f;
        
        // Epoch
        String epochText = String.format("Epoch: %d / %d", service.getCurrentEpoch() + 1, service.getTotalEpochs());
        metricsFont.draw(spriteBatch, epochText, panel.x + padding, y);
        
        // Batch
        y -= lineHeight;
        String batchText = String.format("Batch: %d / %d", service.getCurrentBatch(), service.getTotalBatches());
        metricsFont.draw(spriteBatch, batchText, panel.x + padding, y);
        
        // Loss
        y -= lineHeight;
        metricsFont.setColor(VALUE_COLOR);
        String lossText = String.format("Loss: %.6f", currentLoss);
        metricsFont.draw(spriteBatch, lossText, panel.x + padding, y);
        metricsFont.setColor(TEXT_COLOR);
        
        // Accuracy
        y -= lineHeight;
        String accuracyText = String.format("Accuracy: %.2f%%", service.getCurrentAccuracy() * 100f);
        metricsFont.draw(spriteBatch, accuracyText, panel.x + padding, y);
        
        // Learning Rate
        y -= lineHeight;
        String lrText = String.format("Learning Rate: %.6f", service.getCurrentLearningRate());
        metricsFont.draw(spriteBatch, lrText, panel.x + padding, y);
        
        // Status
        y -= lineHeight;
        String status = service.isPaused() ? "PAUSED" : "RUNNING";
        Color statusColor = service.isPaused() ? WARNING_COLOR : VALUE_COLOR;
        metricsFont.setColor(statusColor);
        metricsFont.draw(spriteBatch, "Status: " + status, panel.x + padding, y);
        metricsFont.setColor(TEXT_COLOR);
    }
    
    private void drawNetworkPanel(VisualizationService service, Rectangle panel) {
        // Draw panel background
        shapeRenderer.setColor(BACKGROUND_COLOR);
        shapeRenderer.rect(panel.x, panel.y, panel.width, panel.height);
        
        // Draw panel border
        shapeRenderer.end();
        shapeRenderer.begin(ShapeType.Line);
        shapeRenderer.setColor(BORDER_COLOR);
        shapeRenderer.rect(panel.x, panel.y, panel.width, panel.height);
        shapeRenderer.end();
        shapeRenderer.begin(ShapeType.Filled);
        
        // Draw title
        spriteBatch.enableBlending();
        titleFont.draw(spriteBatch, "=== NETWORK INFO ===", 
            panel.x + padding, panel.y + panel.height - padding * 1.5f);
        
        // Draw network info
        float y = panel.y + panel.height - padding * 3.5f;
        final float lineHeight = 22f;
        final float yStart = y;
        
        service.getNetworkArchitecture().ifPresent(architecture -> {
            float yPos = yStart;
            String nameText = "Name: " + architecture.name;
            metricsFont.draw(spriteBatch, nameText, panel.x + padding, yPos);
            
            yPos -= lineHeight;
            String layersText = "Layers: " + architecture.getLayerCount();
            metricsFont.draw(spriteBatch, layersText, panel.x + padding, yPos);
            
            yPos -= lineHeight;
            String paramsText = String.format("Parameters: %d", architecture.totalParameters);
            metricsFont.draw(spriteBatch, paramsText, panel.x + padding, yPos);
            
            yPos -= lineHeight;
            String inputText = String.format("Input Size: %d", architecture.inputSize);
            metricsFont.draw(spriteBatch, inputText, panel.x + padding, yPos);
            
            yPos -= lineHeight;
            String outputText = String.format("Output Size: %d", architecture.outputSize);
            metricsFont.draw(spriteBatch, outputText, panel.x + padding, yPos);
        });
        
        // Draw layer legend
        float legendY = yStart - lineHeight * 2;
        smallFont.draw(spriteBatch, "Layer Types:", panel.x + padding, y);
        
        y -= 18;
        drawLayerLegendItem(panel.x + padding, legendY, "IN", NetworkRenderer.getLayerColor(LayerSnapshot.LayerType.INPUT));
        drawLayerLegendItem(panel.x + 70, legendY, "CONV", NetworkRenderer.getLayerColor(LayerSnapshot.LayerType.CONV2D));
        drawLayerLegendItem(panel.x + 140, legendY, "LSTM", NetworkRenderer.getLayerColor(LayerSnapshot.LayerType.BILSTM));
        
        legendY -= 18;
        drawLayerLegendItem(panel.x + padding, legendY, "FC", NetworkRenderer.getLayerColor(LayerSnapshot.LayerType.DENSE));
        drawLayerLegendItem(panel.x + 70, legendY, "OUT", NetworkRenderer.getLayerColor(LayerSnapshot.LayerType.OUTPUT));
    }
    
    private void drawLayerLegendItem(float x, float y, String label, Color color) {
        // Draw color box
        shapeRenderer.setColor(color);
        shapeRenderer.rect(x, y - 12, 12, 12);
        
        // Draw label
        smallFont.setColor(TEXT_COLOR);
        smallFont.draw(spriteBatch, label, x + 16, y);
    }
    
    private void drawControlsPanel(Rectangle panel) {
        // Draw panel background
        shapeRenderer.setColor(BACKGROUND_COLOR);
        shapeRenderer.rect(panel.x, panel.y, panel.width, panel.height);
        
        // Draw panel border
        shapeRenderer.end();
        shapeRenderer.begin(ShapeType.Line);
        shapeRenderer.setColor(BORDER_COLOR);
        shapeRenderer.rect(panel.x, panel.y, panel.width, panel.height);
        shapeRenderer.end();
        shapeRenderer.begin(ShapeType.Filled);
        
        // Draw title
        spriteBatch.enableBlending();
        titleFont.draw(spriteBatch, "=== CONTROLS ===", 
            panel.x + padding, panel.y + panel.height - padding * 1.5f);
        
        // Draw controls
        float y = panel.y + panel.height - padding * 3.5f;
        float lineHeight = 22f;
        
        smallFont.draw(spriteBatch, "SPACE - Pause/Resume", panel.x + padding, y);
        y -= lineHeight;
        smallFont.draw(spriteBatch, "R - Reset View", panel.x + padding, y);
        y -= lineHeight;
        smallFont.draw(spriteBatch, "Mouse Drag - Rotate", panel.x + padding, y);
        y -= lineHeight;
        smallFont.draw(spriteBatch, "Scroll - Zoom", panel.x + padding, y);
        y -= lineHeight;
        smallFont.draw(spriteBatch, "ESC - Exit", panel.x + padding, y);
    }
    
    private void drawLossGraph(Rectangle area) {
        // Draw graph background
        shapeRenderer.setColor(BACKGROUND_COLOR);
        shapeRenderer.rect(area.x, area.y, area.width, area.height);
        
        // Draw graph border
        shapeRenderer.end();
        shapeRenderer.begin(ShapeType.Line);
        shapeRenderer.setColor(BORDER_COLOR);
        shapeRenderer.rect(area.x, area.y, area.width, area.height);
        shapeRenderer.end();
        shapeRenderer.begin(ShapeType.Filled);
        
        // Draw title
        spriteBatch.enableBlending();
        smallFont.draw(spriteBatch, "Loss History", 
            area.x + padding, area.y + area.height - padding);
        
        // Draw loss curve
        if (lossHistory.size() < 2) return;
        
        float graphX = area.x + padding;
        float graphY = area.y + padding;
        float graphWidth = area.width - padding * 2;
        float graphHeight = area.height - padding * 2;
        
        // Find min and max loss values
        float minLoss = Float.MAX_VALUE;
        float maxLoss = Float.MIN_VALUE;
        for (float loss : lossHistory) {
            if (loss < minLoss) minLoss = loss;
            if (loss > maxLoss) maxLoss = loss;
        }
        
        if (maxLoss - minLoss < 0.0001f) maxLoss = minLoss + 1f;
        
        // Draw loss line
        shapeRenderer.end();
        shapeRenderer.begin(ShapeType.Line);
        shapeRenderer.setColor(VALUE_COLOR);
        
        float stepX = graphWidth / (lossHistory.size() - 1);
        
        for (int i = 0; i < lossHistory.size() - 1; i++) {
            float x1 = graphX + i * stepX;
            float y1 = graphY + ((lossHistory.get(i) - minLoss) / (maxLoss - minLoss)) * graphHeight;
            
            float x2 = graphX + (i + 1) * stepX;
            float y2 = graphY + ((lossHistory.get(i + 1) - minLoss) / (maxLoss - minLoss)) * graphHeight;
            
            shapeRenderer.line(x1, y1, x2, y2);
        }
        
        shapeRenderer.end();
        shapeRenderer.begin(ShapeType.Filled);
    }
    
    public void dispose() {
        // Dispose fonts
        if (titleFont != null) titleFont.dispose();
        if (metricsFont != null) metricsFont.dispose();
        if (smallFont != null) smallFont.dispose();
        
        // Dispose rendering resources
        if (spriteBatch != null) spriteBatch.dispose();
        if (shapeRenderer != null) shapeRenderer.dispose();
    }
}
