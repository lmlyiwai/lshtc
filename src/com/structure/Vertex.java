package com.structure;

/**
 * �ṹ����ͼ�еĽڵ�
 * */
public class Vertex {
	private int 		id;				//ÿ���ڵ��ӦΨһID
	private String 		info;			//���������Ϣ
	private int 		level;			//�ڵ������νṹ�е���ȣ���δʵ�֡�
	
	public VertexPoint 	prior;			//ָ�򸸽ڵ������
	public VertexPoint 	next;			//ָ���ӽڵ������
	
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
