package com.apexocr.visualization;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g3d.Model;
import com.badlogic.gdx.graphics.g3d.Renderable;
import com.badlogic.gdx.math.Matrix4;
import com.badlogic.gdx.utils.Array;

/**
 * Represents a connection between two neural network layers in 3D space.
 * Handles connection visualization and flow animations.
 */
public class Connection3D {
    
    private final String name;
    private Model model;
    private final Matrix4 transform = new Matrix4();
    private Color color = new Color(0.5f, 0.5f, 0.5f, 0.3f);
    private float flowIntensity = 0f;
    private float flowDirection = 1f; // 1 = forward, -1 = backward
    private float animationOffset = 0f;
    
    public Connection3D(String name) {
        this.name = name;
        this.transform.idt();
    }
    
    /**
     * Set the 3D model for this connection.
     * @param model LibGDX Model
     */
    public void setModel(Model model) {
        this.model = model;
    }
    
    /**
     * Set the position of the connection in 3D space.
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     */
    public void setPosition(float x, float y, float z) {
        transform.setToTranslation(x, y, z);
    }
    
    /**
     * Set the flow intensity for animation.
     * @param intensity Flow intensity (0-1)
     */
    public void setFlowIntensity(float intensity) {
        this.flowIntensity = Math.max(0f, Math.min(1f, intensity));
    }
    
    /**
     * Set the flow direction.
     * @param direction 1 for forward pass, -1 for backward pass
     */
    public void setFlowDirection(float direction) {
        this.flowDirection = Math.signum(direction);
    }
    
    /**
     * Set the color of the connection.
     * @param color Color to set
     */
    public void setColor(Color color) {
        this.color = color;
    }
    
    /**
     * Update animation state.
     * @param delta Time since last update
     */
    public void update(float delta) {
        animationOffset += delta * flowDirection * flowIntensity * 2f;
        if (animationOffset > 1f) animationOffset -= 1f;
        if (animationOffset < 0f) animationOffset += 1f;
    }
    
    /**
     * Get all renderables from this connection.
     * @param renderables Array to add renderables to
     */
    public void getRenderables(Array<Renderable> renderables) {
        if (model != null) {
            // Apply flow animation to opacity/color
            float animatedAlpha = 0.3f + flowIntensity * 0.7f;
            
            if (model.nodes != null && !model.nodes.isEmpty()) {
                model.nodes.get(0).parts.forEach(part -> {
                    if (part.material != null) {
                        part.material.setColorAttribute(
                            com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute.createDiffuse(
                                new Color(color.r, color.g, color.b, animatedAlpha)
                            )
                        );
                    }
                });
            }
            
            model.getRenderables(renderables, transform);
        }
    }
    
    /**
     * Dispose of resources.
     */
    public void dispose() {
        if (model != null) {
            model.dispose();
        }
    }
    
    /**
     * Get the connection name.
     * @return Connection name
     */
    public String getName() {
        return name;
    }
    
    /**
     * Get the current flow intensity.
     * @return Flow intensity
     */
    public float getFlowIntensity() {
        return flowIntensity;
    }
}
