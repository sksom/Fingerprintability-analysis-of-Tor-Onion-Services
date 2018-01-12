for file in /home/nithin/Desktop/BigData_Project/140203_042843/0/0-google.de/*/*.pcap
do
	"tshark -r "$file" -T fields -e frame.number -e eth.src -e eth.dst -e ip.src -e ip.dst -e frame.protocols -e frame.len -e tcp.analysis.ack_rtt -E header=y -E seperator=," > "test.csv" 
done
