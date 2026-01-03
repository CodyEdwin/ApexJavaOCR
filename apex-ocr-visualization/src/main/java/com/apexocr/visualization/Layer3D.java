package com.apexocr.visualization;

import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.g3d.Model;
import com.badlogic.gdx.graphics.g3d.Renderable;
import com.badlogic.gdx.graphics.g3d.model.Node;
import com.badlogic.gdx.math.Matrix4;
import com.badlogic.gdx.utils.Array;

/**
 * Represents a single neural network layer in 3D space.
 * Handles layer model, position, scale, and color.
 */
public class Layer3D {
    
    private final String name;
    private Model model;
    private final Node rootNode;
    private final Matrix4 transform = new Matrix4();
    private Color color = Color.WHITE;
    private float scale = 1f;
    
    public Layer3D(String name) {
        this.name = name;
        this.rootNode = new Node();
        this.transform.idt();
    }
    
    /**
     * Set the 3D model for this layer.
     * @param model LibGDX Model
     */
    public void setModel(Model model) {
        this.model = model;
        if (model != null) {
            rootNode.addChild(model.nodes.get(0));
        }
    }
    
    /**
     * Set the position of the layer in 3D space.
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     */
    public void setPosition(float x, float y, float z) {
        transform.setToTranslation(x, y, z);
        updateTransform();
    }
    
    /**
     * Set the scale of the layer.
     * @param scale Scale factor
     */
    public void setScale(float scale) {
        this.scale = scale;
        updateTransform();
    }
    
    /**
     * Set the color of the layer.
     * @param color Color to set
     */
    public void setColor(Color color) {
        this.color = color;
        if (model != null) {
            model.nodes.get(0).parts.forEach(part -> {
                if (part.material != null) {
                    part.material.setColorAttribute(
                        com.badlogic.gdx.graphics.g3d.attributes.ColorAttribute.createDiffuse(color)
                    );
                }
            });
        }
    }
    
    /**
     * Get all renderables from this layer.
     * @param renderables Array to add renderables to
     */
    public void getRenderables(Array<Renderable> renderables) {
        if (model != null) {
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
     * Get the layer name.
     * @return Layer name
     */
    public String getName() {
        return name;
    }
    
    /**
     * Update the transform matrix based on position and scale.
     */
    private void updateTransform() {
        transform.scl(scale);
        if (rootNode != null) {
            rootNode.localTransform.set(transform);
        }
    }
    
    /**
     * Get the current transform.
     * @return Transform matrix
     */
    public Matrix4 getTransform() {
        return new Matrix4(transform);
    }
}
