package com.structure;

/**
 * �ڽӱ���ָ�����ӽڵ�򸸽ڵ�
 * */
public class VertexPoint {
	public int			offset;
	public VertexPoint 	next;
	
	public VertexPoint(int offset) {
		this.offset = offset;
		this.next = null;
	}
}
