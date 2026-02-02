package com.ripple;

/**
 * Represents an anchor point within a trajectory at a specific temporal frame.
 * An anchor point defines a spatial-temporal constraint used in trajectory optimization.
 */
public class Anchor {
    public final int frame;  // Zero-indexed temporal frame identifier
    public final int x;       // Horizontal pixel coordinate
    public final int y;       // Vertical pixel coordinate
    
    public Anchor(int frame, int x, int y) {
        this.frame = frame;
        this.x = x;
        this.y = y;
    }
    
    public Anchor(Anchor other) {
        this.frame = other.frame;
        this.x = other.x;
        this.y = other.y;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Anchor)) return false;
        Anchor other = (Anchor) obj;
        return frame == other.frame && x == other.x && y == other.y;
    }
    
    @Override
    public int hashCode() {
        return frame * 31 * 31 + x * 31 + y;
    }
    
    @Override
    public String toString() {
        return String.format("Anchor(frame=%d, x=%d, y=%d)", frame, x, y);
    }
}