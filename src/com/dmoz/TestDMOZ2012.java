package com.dmoz;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

import com.structure.Structure;

public class TestDMOZ2012 {

	@Test
	public void test() throws IOException {
		String cfile = "F:\\DataSets\\Dmoz\\DMOZ2012\\track2-DMOZ-hierarchy.txt";
		Map<Integer, Integer> map = ReadData.readDMOZ2012Map(cfile);
		Map<Integer, Integer> rmap = ReadData.reverseMap(map);
		Structure tree = ReadData.getDMOZ2012Structure(cfile, map);
		Map<Integer, int[]> pathes = tree.getAllPath();
		
		int[] nodes = tree.getLeaves();
		double dis = tree.getDistance(nodes[1], nodes[2]);
		String outfile = "F:\\DataSets\\Dmoz\\DMOZ2012\\pathes.txt";
		BufferedWriter out = new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(outfile)));
		out.write(rmap.get(nodes[1]) + " " + rmap.get(nodes[2]) + " " + dis + "\n");
		Set<Integer> key = pathes.keySet();
		Iterator<Integer> it = key.iterator();
		while(it.hasNext()) {
			String line = new String();
			int[] path = pathes.get(it.next());
			for(int i = 0; i < path.length; i++) {
				line += rmap.get(path[i]) + " ";
			}
			line += "\n";
			out.write(line);
		}
		
		String line = new String();
		int root = tree.getRoot();
		line += root;
		out.write(line);
		out.close();
	}

}
