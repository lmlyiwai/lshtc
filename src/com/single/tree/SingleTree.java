package com.single.tree;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class SingleTree {
	//读取节点
	public static String[][] readLabelPairs(String filename) throws IOException {
		BufferedReader in = new BufferedReader(
				new InputStreamReader(new FileInputStream(filename)));
		String line = null;
		List<String> parentList = new ArrayList<String>();
		List<String> childList = new ArrayList<String>();
		
		String[] splits = null;
		while((line = in.readLine()) != null) {
			splits = line.split("\\s|:\\s|\t|\t|\n");
			
			//读取C目录
			if(!splits[1].equals("None") && splits[1].startsWith("C")) {
				parentList.add(splits[1].toUpperCase());
				childList.add(splits[3].toUpperCase());
			}
		}
		
		String[][] result = new String[parentList.size()][2];
		for(int i = 0; i < parentList.size(); i++) {
			result[i][0] = parentList.get(i);
			result[i][1] = childList.get(i);
		}
		return result;
	} 
	
	//给每个类别指定一个数字ID
		public static Map<String, Integer> getIDLabelPair(String[][] pairs) {
			Set<String> set = new HashSet<String>();
			for(int i = 0; i < pairs.length; i++) {
				set.add(pairs[i][0]);
				set.add(pairs[i][1]);
			}
			
			Map<String, Integer> map = new HashMap<String, Integer>();
			
			Iterator<String> it = set.iterator();
			String str;
			int counter = 0;
			while(it.hasNext()) {
				str = it.next();
				map.put(str, counter++);
			}
			return map;
		}
		
		
}
