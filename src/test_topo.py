from mininet.net import Mininet
from mininet.cli import CLI

net = Mininet()

h1 = net.addHost('h1')
h2 = net.addHost('h2')

s1 = net.addSwitch('s1')

net.addLink(h1, s1)
net.addLink(h2, s1)

net.start()

CLI(net)

net.stop()
