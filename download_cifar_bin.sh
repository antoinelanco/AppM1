
#!/bin/bash
if ! [ -d "./cifar-10-batches-bin"]; then
	wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
	tar xvzf cifar-10-binary.tar.gz
	rm -f cifar-10-binary.tar.gz
fi
