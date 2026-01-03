package com.apexocr.visualization;

import com.apexocr.training.monitoring.VisualizationService;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.GlyphLayout;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.g2d.freetype.FreeTypeFontGenerator;
import com.badlogic.gdx.graphics.g2d.freetype.FreeTypeFontGenerator.FreeTypeFontParameter;
import com.badlogic.gdx.utils.Array;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

/**
 * 2D HUD overlay for displaying training metrics and controls.
 * Renders on top of the 3D scene using SpriteBatch.
 */
public class HUDOverlay {
    
    private final SpriteBatch spriteBatch;
    private BitmapFont titleFont;
    private BitmapFont metricsFont;
    private BitmapFont smallFont;
    
    private final DecimalFormat lossFormat = new DecimalFormat("0.000000");
    private final DecimalFormat percentFormat = new DecimalFormat("0.00");
    private final DecimalFormat timeFormat = new DecimalFormat("0.0");
    
    private int screenWidth;
    private int screenHeight;
    
    // UI Elements
    private final List<MetricDisplay> metricDisplays = new ArrayList<>();
    private final List<GraphDisplay> graphDisplays = new ArrayList<>();
    
    // Colors
    private static final Color BACKGROUND_COLOR = new Color(0.12f, 0.13f, 0.14f, 0.85f);
    private static final Color TEXT_COLOR = new Color(0.95f, 0.95f, 0.95f, 1f);
    private static final Color ACCENT_COLOR = new Color(0.23f, 0.51f, 0.96f, 1f);
    private static final Color SUCCESS_COLOR = new Color(0.10f, 0.71f, 0.51f, 1f);
    private static final Color WARNING_COLOR = new Color(0.96f, 0.76f, 0.03f, 1f);
    private static final Color ERROR_COLOR = new Color(0.93f, 0.27f, 0.27f, 1f);
    
    public HUDOverlay() {
        this.spriteBatch = new SpriteBatch();
        initializeFonts();
        initializeMetricDisplays();
    }
    
    /**
     * Initialize fonts for the HUD.
     */
    private void initializeFonts() {
        FreeTypeFontGenerator generator = new FreeTypeFontGenerator(Gdx.files.internal("fonts/Roboto-Regular.ttf"));
        
        // Title font
        FreeTypeFontParameter titleParams = new FreeTypeFontParameter();
        titleParams.size = 24;
        titleParams.color = TEXT_COLOR;
        titleFont = generator.generateFont(titleParams);
        
        // Metrics font
        FreeTypeFontParameter metricsParams = new FreeTypeFontParameter();
        metricsParams.size = 18;
        metricsParams.color = TEXT_COLOR;
        metricsFont = generator.generateFont(metricsParams);
        
        // Small font
        FreeTypeFontParameter smallParams = new FreeTypeFontParameter();
        smallParams.size = 14;
        smallParams.color = new Color(0.7f, 0.7f, 0.7f, 1f);
        smallFont = generator.generateFont(smallParams);
        
        generator.dispose();
    }
    
    /**
     * Initialize metric display positions.
     */
    private void initializeMetricDisplays() {
        // Metrics will be populated during resize
    }
    
    /**
     * Handle window resize.
     * @param width New width
     * @param height New height
     */
    public void resize(int width, int height) {
        this.screenWidth = width;
        this.screenHeight = height;
        
        // Update metric display positions
        metricDisplays.clear();
        
        // Top-left metrics
        metricDisplays.add(new MetricDisplay("Loss", 20, height - 40, MetricType.LOSS));
        metricDisplays.add(new MetricDisplay("Accuracy", 20, height - 70, MetricType.ACCURACY));
        metricDisplays.add(new MetricDisplay("Epoch", 20, height - 100, MetricType.EPOCH));
        metricDisplays.add(new MetricDisplay("Batch", 20, height - 130, MetricType.BATCH));
        metricDisplays.add(new MetricDisplay("LR", 20, height - 160, MetricType.LEARNING_RATE));
        
        // Memory usage
        metricDisplays.add(new MetricDisplay("Memory", 20, height - 200, MetricType.MEMORY));
        
        // Progress bar
        metricDisplays.add(new MetricDisplay("Progress", 20, height - 240, MetricType.PROGRESS));
    }
    
    /**
     * Render the HUD overlay.
     * @param service Visualization service providing training data
     */
    public void render(VisualizationService service) {
        spriteBatch.begin();
        
        // Draw control panel background
        drawControlPanel();
        
        // Draw metrics
        drawMetrics(service);
        
        // Draw progress bar
        drawProgressBar(service);
        
        // Draw loss graph (simplified)
        drawLossGraph(service);
        
        // Draw training controls hint
        drawControlsHint();
        
        spriteBatch.end();
    }
    
    /**
     * Draw the control panel background.
     */
    private void drawControlPanel() {
        // Left panel background
        spriteBatch.setColor(BACKGROUND_COLOR);
        float panelWidth = 280f;
        spriteBatch.draw(
            com.badlogic.gdx.graphics.TextureUtils.createWhiteTexture(),
            0, 0, panelWidth, screenHeight
        );
        
        // Separator line
        spriteBatch.setColor(ACCENT_COLOR);
        spriteBatch.draw(
            com.badlogic.gdx.graphics.TextureUtils.createWhiteTexture(),
            panelWidth - 2, 0, 2, screenHeight
        );
    }
    
    /**
     * Draw all metrics.
     * @param service Visualization service
     */
    private void drawMetrics(VisualizationService service) {
        float currentLoss = service.getCurrentLoss();
        float currentAccuracy = service.getCurrentAccuracy();
        int epoch = service.getCurrentEpoch();
        int totalEpochs = service.getTotalEpochs();
        int batch = service.getCurrentBatch();
        int totalBatches = service.getTotalBatches();
        float lr = service.getCurrentLearningRate();
        
        // Draw title
        titleFont.draw(spriteBatch, "Apex OCR Training", 20, screenHeight - 20);
        
        // Draw metrics
        metricsFont.draw(spriteBatch, 
            "Loss: " + lossFormat.format(currentLoss), 20, screenHeight - 55);
        
        metricsFont.draw(spriteBatch, 
            "Accuracy: " + percentFormat.format(currentAccuracy * 100) + "%", 20, screenHeight - 85);
        
        metricsFont.draw(spriteBatch, 
            "Epoch: " + (epoch + 1) + " / " + totalEpochs, 20, screenHeight - 115);
        
        metricsFont.draw(spriteBatch, 
            "Batch: " + (batch + 1) + " / " + totalBatches, 20, screenHeight - 145);
        
        metricsFont.draw(spriteBatch, 
            "Learning Rate: " + String.format("%.6f", lr), 20, screenHeight - 175);
    }
    
    /**
     * Draw the progress bar.
     * @param service Visualization service
     */
    private void drawProgressBar(VisualizationService service) {
        float progress = service.getProgress();
        int epoch = service.getCurrentEpoch();
        int totalEpochs = service.getTotalEpochs();
        
        // Progress bar background
        float barX = 20f;
        float barY = screenHeight - 230f;
        float barWidth = 240f;
        float barHeight = 20f;
        
        spriteBatch.setColor(0.3f, 0.3f, 0.3f, 1f);
        spriteBatch.draw(
            com.badlogic.gdx.graphics.TextureUtils.createWhiteTexture(),
            barX, barY, barWidth, barHeight
        );
        
        // Progress bar fill
        if (progress > 0) {
            float fillWidth = barWidth * progress;
            
            // Color based on progress
            Color fillColor = new Color(0.23f, 0.51f, 0.96f, 1f);
            if (progress > 0.5f) fillColor = new Color(0.10f, 0.71f, 0.51f, 1f);
            if (progress > 0.8f) fillColor = new Color(0.96f, 0.76f, 0.03f, 1f);
            
            spriteBatch.setColor(fillColor);
            spriteBatch.draw(
                com.badlogic.gdx.graphics.TextureUtils.createWhiteTexture(),
                barX, barY, fillWidth, barHeight
            );
        }
        
        // Progress text
        smallFont.draw(spriteBatch, 
            percentFormat.format(progress * 100) + "% Complete", barX, barY - 5);
    }
    
    /**
     * Draw a simplified loss graph.
     * @param service Visualization service
     */
    private void drawLossGraph(VisualizationService service) {
        float graphX = 20f;
        float graphY = 180f;
        float graphWidth = 240f;
        float graphHeight = 120f;
        
        // Graph background
        spriteBatch.setColor(new Color(0.15f, 0.16f, 0.17f, 0.9f));
        spriteBatch.draw(
            com.badlogic.gdx.graphics.TextureUtils.createWhiteTexture(),
            graphX, graphY, graphWidth, graphHeight
        );
        
        // Graph border
        spriteBatch.setColor(ACCENT_COLOR);
        spriteBatch.draw(
            com.badlogic.gdx.graphics.TextureUtils.createWhiteTexture(),
            graphX, graphY, graphWidth, 2
        );
        spriteBatch.draw(
            com.badlogic.gdx.graphics.TextureUtils.createWhiteTexture(),
            graphX, graphY + graphHeight - 2, graphWidth, 2
        );
        
        // Graph title
        smallFont.draw(spriteBatch, "Loss Over Time", graphX + 10, graphY + graphHeight - 10);
        
        // Draw simple loss curve (placeholder - would need real data)
        spriteBatch.setColor(SUCCESS_COLOR);
        float lineX1 = graphX + 20;
        float lineY1 = graphY + graphHeight - 40;
        float lineX2 = graphX + graphWidth - 20;
        float lineY2 = graphY + 30;
        
        // Draw a simple curve
        for (int i = 0; i < graphWidth - 40; i++) {
            float t = (float) i / (graphWidth - 40);
            float y = (float) (graphY + graphHeight - 40 - t * (graphHeight - 70) + 
                    Math.sin(t * Math.PI * 4) * 10);
            
            if (i > 0) {
                float prevY = (float) (graphY + graphHeight - 40 - ((t - 0.02f) * (graphHeight - 70)) + 
                        Math.sin((t - 0.02f) * Math.PI * 4) * 10);
                
                com.badlogic.gdx.graphics.glutils.ShapeRenderer shapeRenderer = 
                    new com.badlogic.gdx.graphics.glutils.ShapeRenderer();
                shapeRenderer.begin(com.badlogic.gdx.graphics.glutils.ShapeRenderer.ShapeType.Line);
                shapeRenderer.setColor(SUCCESS_COLOR);
                shapeRenderer.line(lineX1 + i - 1, prevY, lineX1 + i, y);
                shapeRenderer.end();
            }
        }
    }
    
    /**
     * Draw keyboard controls hint.
     */
    private void drawControlsHint() {
        float hintY = 40f;
        
        smallFont.draw(spriteBatch, "Controls:", 20, hintY);
        smallFont.draw(spriteBatch, "SPACE - Pause/Resume", 20, hintY - 25);
        smallFont.draw(spriteBatch, "R - Reset Camera", 20, hintY - 45);
        smallFont.draw(spriteBatch, "Mouse Drag - Rotate View", 20, hintY - 65);
        smallFont.draw(spriteBatch, "Scroll - Zoom In/Out", 20, hintY - 85);
        smallFont.draw(spriteBatch, "ESC - Exit", 20, hintY - 105);
    }
    
    /**
     * Dispose of resources.
     */
    public void dispose() {
        spriteBatch.dispose();
        titleFont.dispose();
        metricsFont.dispose();
        smallFont.dispose();
    }
    
    /**
     * Metric type enumeration.
     */
    private enum MetricType {
        LOSS,
        ACCURACY,
        EPOCH,
        BATCH,
        LEARNING_RATE,
        MEMORY,
        PROGRESS
    }
    
    /**
     * Simple metric display container.
     */
    private static class MetricDisplay {
        final String label;
        final float x, y;
        final MetricType type;
        
        MetricDisplay(String label, float x, float y, MetricType type) {
            this.label = label;
            this.x = x;
            this.y = y;
            this.type = type;
        }
    }
    
    /**
     * Simple graph display container.
     */
    private static class GraphDisplay {
        final String title;
        final float x, y, width, height;
        
        GraphDisplay(String title, float x, float y, float width, float height) {
            this.title = title;
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
        }
    }
}
