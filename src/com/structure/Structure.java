package com.structure;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

/**
 * ���ṹ���ڵ���0��totleVertex-1��������ʾ
 * */

public class Structure {
	private int 					totleVertex;				//���ṹ�нڵ�����
	private Vertex[] 				vertex;						//���ṹ�еĽڵ�
	private Map<Integer, Integer> 	innerToAdd;     //Ϊ�м�ڵ����Ҷ�ڵ㣬��map����м�ڵ�ID��Ӧ��Ҷ�ڵ�ID
	
	public Structure(int totleVertex) {
		this.totleVertex = totleVertex;
		this.vertex = new Vertex[totleVertex];
		for(int i = 0; i < this.vertex.length; i++) {
			this.vertex[i] = new Vertex(i);
		}
	}
	
	/**
	 * ��ӽڵ�,�Խڵ�Ե���ʽ����.
	 * */
	public void addChild(int parent, int child) {
		if(vertex[parent] == null) {
			vertex[parent] = new Vertex(parent);
		}
		
		if(vertex[child] == null) {
			vertex[child] = new Vertex(child);
		}
		
		int[] children = this.getChildren(parent);
		
		if(contain(children, child)) {
			return;
		}

		VertexPoint c = new VertexPoint(child);
		c.next = vertex[parent].next;
		vertex[parent].next = c;
		
		VertexPoint p = new VertexPoint(parent);
		p.next = vertex[child].prior;
		vertex[child].prior = p;
	}
	
	/**
	 * ���ĳ���ڵ���ӽڵ�
	 * */
	public int[] getChildren(int id) {
		if(id >= this.totleVertex) {
			return null;
		}
		
		Vertex v = this.vertex[id];
		List<Integer> list = new ArrayList<Integer>();
		VertexPoint vp = v.next;
		
		while(vp != null) {
			list.add(vp.offset);
			vp = vp.next;
		}
		
		int[] result = new int[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}

	/**
	 * ���ĳ���ڵ�����и��ڵ�
	 * */
	public int[] getParents(int id) {
		if(id >= this.totleVertex) {
			return null;
		}
		
		List<Integer> list = new ArrayList<Integer>();
		Vertex v = this.vertex[id];
		VertexPoint vp = v.prior;
		
		while(vp != null) {
			list.add(vp.offset);
			vp = vp.next;
		}
		
		int[] result = new int[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}

	
	/**
	 * ���ĳ���ڵ��һ�����ڵ�
	 * */
	public int getParent(int id) {
		int[] parents = getParents(id);
		if(parents != null && parents.length != 0) {
			return parents[0];
		} else {
			return -1;
		}
	}
	
	/**
	 * ���ĳһ�ڵ㵽���ڵ��·��,��ʱֻʹ�������νṹ��ͼ�Ļ������ڶ�·���е�ĳһ��·��
	 * */
	public int[] getPathToRoot(int id) {
		Vertex v = this.vertex[id];
		VertexPoint vp = null;
		
		int idnext;
		List<Integer> path = new ArrayList<Integer>();
		path.add(id);
		while(v.prior != null) {
			vp = v.prior;
			idnext = vp.offset;
			path.add(idnext);
			v = this.vertex[idnext];
		}
		int[] result = new int[path.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = path.get(i);
		}
		return result;
	}
	
	/**
	 * ���������ӽڵ���
	 * */
	public int[] getLeaves() {
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i < this.vertex.length; i++) {
			if(this.vertex[i].next == null) {
				list.add(i);
			}
		}
		
		int[] result = new int[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	/**
	 * ���������м�ڵ�
	 **/
	public int[] getInnerVertex() {
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i < this.vertex.length; i++) {
			if(this.vertex[i].next != null) {
				list.add(i);
			}
		}
		
		int[] result = new int[list.size()];
		for(int i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}

	public int getTotleVertex() {
		return totleVertex;
	}

	public void setTotleVertex(int totleVertex) {
		this.totleVertex = totleVertex;
	}

	public Vertex[] getVertex() {
		return vertex;
	}

	public void setVertex(Vertex[] vertex) {
		this.vertex = vertex;
	}
	
	public int[] getAllNodes() {
		int[] result = new int[this.vertex.length];
		for(int i = 0; i < result.length; i++) {
			result[i] = i;
		}
		return result;
	}
	
	public boolean isLeaf(int id) {
		Vertex v = this.vertex[id];
		if(v.next == null) {
			return true;
		} else {
			return false;
		}
	}
	
	public int getRoot() {
		int root = -1;
		for(int i = 0; i < this.totleVertex; i++) {
			if(this.vertex[i].prior == null) {
				root = i;
				return root;
			}
		}
		return root;
	}
	
	/**
	 * �������array�Ƿ����value.
	 * */
	public boolean contain(int[] array, int value) {
		boolean flag = false;
		
		if(array == null) {
			return flag;
		}
		
		for(int i = 0; i < array.length; i++) {
			if(array[i] == value) {
				flag = true;
				break;
			}
		}
		
		return flag;
	}
	
	/**
	 * ��չ�ṹ��Ϊ���ṹ��ÿһ���ڵ����һ��Ҷ�ڵ�
	 * */
	public void extendStructure() {
		int oldTotleVertex = this.getTotleVertex();
		int[] innerNode = this.getInnerVertex();
		int root = this.getRoot();
		
		this.setTotleVertex(oldTotleVertex + innerNode.length - 1);
//		this.setTotleVertex(oldTotleVertex + innerNode.length);
		
		int i;
		Vertex[] newV = new Vertex[this.totleVertex];
		for(i = 0; i < oldTotleVertex; i++) {
			newV[i] = this.vertex[i];
		}
		
		this.vertex = newV;
		
		while(i < oldTotleVertex + innerNode.length - 1) {  //-1
			this.vertex[i++] = new Vertex(i);
		}
				
		int startID = oldTotleVertex;
		int par;
		int chi;
		
		Map<Integer,Integer> map = new HashMap<Integer, Integer>();
		
		for(i = 0; i < innerNode.length; i++) {
			par = innerNode[i];
			
			if(par != this.getRoot()) {
				chi = startID++;
				this.addChild(par, chi);
				map.put(par, chi);
			}
			
		}
		this.innerToAdd = map;
	}

	public Map<Integer, Integer> getInnerToAdd() {
		return innerToAdd;
	}

	public void setInnerToAdd(Map<Integer, Integer> innerToAdd) {
		this.innerToAdd = innerToAdd;
	}
	
	/**
	 * �����ĳ�ڵ�Ϊ���ڵ������Ҷ�ڵ�
	 * */
	public int[] getDescendent(int id) {
		List<Integer> list = new ArrayList<Integer>();
		Queue<Integer> queue = new LinkedList<Integer>();
		
		queue.offer(id);
		
		int[] children = null;
		int currentID, i;
		while(queue.size() != 0) {
			currentID = queue.poll();
			if(isLeaf(currentID)) {
				list.add(currentID);
			}
			
			children = getChildren(currentID);
			if(children != null && children.length != 0) {
				for(i = 0; i < children.length; i++) {
					queue.offer(children[i]);
				}
			}
		}
		
		int[] result = new int[list.size()];
		for(i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	/**
	 * �����ĳһ�ڵ�Ϊ�������������нڵ㣬�����ýڵ�����
	 * */
	public int[] getDes(int id) {
		List<Integer> list = new ArrayList<Integer>();
		Queue<Integer> queue = new LinkedList<Integer>();
		
		queue.offer(id);
		
		int[] children = null;
		int currentID, i;
		while(queue.size() != 0) {
			currentID = queue.poll();
			list.add(currentID);
			
			children = getChildren(currentID);
			if(children != null && children.length != 0) {
				for(i = 0; i < children.length; i++) {
					queue.offer(children[i]);
				}
			}
		}
		
		int[] result = new int[list.size()];
		for(i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	/**
	 * ��α�����������ڵ�˳�򣬲��������ڵ�
	 * */
	public int[] levelTraverse() {
		int root = this.getRoot();
		Queue<Integer> queue = new LinkedList<Integer>();
		queue.add(root);
		
		List<Integer> list = new ArrayList<Integer>();
		int current;
		int[] children = null;
		int i;
		while(queue.size() != 0) {
			current = queue.poll();
			children = this.getChildren(current);
			
			if(current != root) {
				list.add(current);
			}
			
			if(children != null && children.length != 0) {
				for(i = 0; i < children.length; i++) {
					queue.offer(children[i]);
				}
			}
		}
		
		int[] result = new int[list.size()];
		for(i = 0; i < result.length; i++) {
			result[i] = list.get(i);
		}
		return result;
	}
	
	/**
	 *	�������ڵ��ľ��� 
	 */
	public double getDistance(int ida, int idb) {
		int[] patha = getPathToRoot(ida);
		int[] pathb = getPathToRoot(idb);
		
		if(patha == null  && pathb != null) {
			return pathb.length;
		}
		
		if(patha != null && pathb == null) {
			return patha.length;
		}
		
		if(patha == null && pathb == null) {
			return 0;
		}
		
		double result = patha.length + pathb.length;
		double common = 0;
		Set<Integer> set = new HashSet<Integer>();
		for(int i = 0; i < patha.length; i++) {
			set.add(patha[i]);
		}
		
		for(int i = 0; i < pathb.length; i++) {
			if(set.contains(pathb[i])) {
				common = common + 1;
			}
		}
		
		result = result - 2 * common;
		return result;
	}
	
	/**
	 * ���Ŀ¼�ṹ�ڵ��������
	 */
	public double getMaxDistance() {
		int[] leaves = getLeaves();
		double maxDistance = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < leaves.length; i++) {
			for(int j = 0; j < leaves.length; j++) {
				int a = leaves[i];
				int b = leaves[j];
				double distance = getDistance(a, b);
				if(distance > maxDistance) {
					maxDistance = distance;
				}
			}
		}
		return maxDistance;
	}
	
	/**
	 * ��ȡ�ṹ������·��
	 */
	public Map<Integer, int[]> getAllPath() {
		int[] leaves = getLeaves();
		Map<Integer, int[]> map = new HashMap<Integer, int[]>();
		for(int i = 0; i < leaves.length; i++) {
			int key = leaves[i];
			int[] value = getPathToRoot(key);
			int[] rvalue = new int[value.length];
			for(int j = 0; j < value.length; j++) {
				rvalue[j] = value[value.length - j - 1];
			}
			map.put(key, rvalue);
		}
		return map;
	}
	
	/**
	 * 
	 */
	public double getDistance(int[] patha, int[] pathb) {
		double result = patha.length + pathb.length;
		double common = 0;
		int tc = 0;
		while(tc < patha.length && tc < pathb.length) {
			if(patha[tc] == pathb[tc]) {
				common++;
			}
			tc++;
		}
		
		result = result - 2 * common;
		return result;
	}
}
