from functools import partial
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.link import TCLink
from mininet.node import OVSSwitch
from mininet.cli import CLI

class BottleneckTopo(Topo):
    def build(self):

        h1 = self.addHost('h1')
        h2 = self.addHost('h2')
        h3 = self.addHost('h3')

        s1 = self.addSwitch('s1')
        s2 = self.addSwitch('s2')

        self.addLink(h1, s1)

        self.addLink(h2, s1)

        self.addLink(h3, s2)

        # Bottleneck Link
        self.addLink(
            s1,
            s2,
            bw=10,
            delay='20ms'
        )

topo = BottleneckTopo()

net = Mininet(
    topo=topo,
    link=TCLink,
    # IMPORTANT: controller=None, but that alone leaves OVSSwitch in
    # failMode='secure' (empty flow table, drops everything). Force
    # 'standalone' so switches behave as plain L2 learning switches.
    switch=partial(OVSSwitch, failMode='standalone'),
    controller=None
)

net.start()

CLI(net)

net.stop()


