package com.structure;

/**
 * 邻接表中指向其子节点或父节点
 * */
public class VertexPoint {
	public int			offset;
	public VertexPoint 	next;
	
	public VertexPoint(int offset) {
		this.offset = offset;
		this.next = null;
	}
}
