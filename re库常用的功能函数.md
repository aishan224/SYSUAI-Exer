# re库常用的功能函数：

## **re.search()**

在一个字符串中搜索匹配正则表达式的第一个位置，返回match对象

re.search(pattern, string, flags=0)

​	pattern:正则表达式的字符串或者原生字符串表示

​	string:  待匹配的字符串

​	flags:   正则表达式使用时的控制标记:

​		re.I re.IGNORECASE: 忽略正则表达式的大小写，[A-Z]能够匹配小写字符

​		re.M re.MULTILINE : 正则表达式中的^操作符能够讲给定字符串的每行当作匹配开始

​		re.S re.DOTALL : 正则表达式中的.操作付能够匹配所有字符， 默认匹配除换行外的所有字符

<a href="http://www.baidu.com">clickhere</a>

## **re.match()**

从一个字符串的开始位置起匹配正则表达式，返回match对象

re.match(pattern, string, flags=0)  # 同上

## **re.findall()**

搜索字符串，以列表形式返回全部能匹配的子串

re.findall(pattern, string, flags=0) # 同上

## **re.split()**

一个字符串按照正在为表达式匹配结果进行分割，返回列表类型

re.split(pattern, string, maxsplit=0, flags=0)# 同上

​	maxsplit：最大分割数，用户约定分割多少个出来，超出的部分直接当成一个整体

## **re.finditer()**

搜索字符串，返回一个匹配结果的迭代类型，每个迭代元素都是match对象

函数参数如上

## **re.sub()**

在一个字符串中替换所有匹配正则表达式的子串，返回替换后的字符串

re.sub(pattern, repl, string, count=0, flags=0)

​	repl: 用来替换匹配到的字符串的字符串

​	count： 匹配的最大替换次数