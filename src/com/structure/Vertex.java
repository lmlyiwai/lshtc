package com.structure;

/**
 * 结构树或图中的节点
 * */
public class Vertex {
	private int 		id;				//每个节点对应唯一ID
	private String 		info;			//保存额外信息
	private int 		level;			//节点在树形结构中的深度，暂未实现。
	
	public VertexPoint 	prior;			//指向父节点的链表
	public VertexPoint 	next;			//指向孩子节点的链表
	
	public Vertex(int id) {
		this.id = id;
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}

	public String getInfo() {
		return info;
	}

	public void setInfo(String info) {
		this.info = info;
	}

	public int getLevel() {
		return level;
	}

	public void setLevel(int level) {
		this.level = level;
	}	
}
