"""E4 GCN+, GAT+ and GraphSAGE+ architecture from expH_gnn2.py."""
EMB = 16

def build(torch, nn, F, conv, pool):
    GCNConv, GATConv, SAGEConv = conv
    gmean, gmax = pool

    class Base(nn.Module):
        """Shared skeleton: per-sensor embedding, three conv layers, mean+max readout."""

        def __init__(self, n_sensors, hidden=128, num_classes=15, drop=0.3):
            super().__init__()
            self.emb = nn.Embedding(n_sensors, EMB)
            self.pre = nn.Linear(1 + EMB, hidden)
            self.bn1, self.bn2 = (nn.BatchNorm1d(hidden), nn.BatchNorm1d(hidden))
            self.fc1 = nn.Linear(hidden * 2 + n_sensors, hidden)
            self.fc2 = nn.Linear(hidden, num_classes)
            self.drop, self.n = (drop, n_sensors)

        def head(self, x, batch, raw):
            x = torch.cat([gmean(x, batch), gmax(x, batch), raw], dim=1)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=self.drop, training=self.training)
            return self.fc2(x)

        def stem(self, data):
            ids = torch.arange(self.n, device=data.x.device).repeat(data.x.shape[0] // self.n)
            return F.relu(self.pre(torch.cat([data.x, self.emb(ids)], dim=1)))

    class GCNp(Base):

        def __init__(self, n_sensors, hidden=128, **kw):
            super().__init__(n_sensors, hidden, **kw)
            self.c1, self.c2, self.c3 = (GCNConv(hidden, hidden), GCNConv(hidden, hidden), GCNConv(hidden, hidden))

        def forward(self, d):
            x, ei = (self.stem(d), d.edge_index)
            x = F.relu(self.bn1(self.c1(x, ei)))
            x = F.dropout(x, p=self.drop, training=self.training)
            x = F.relu(self.bn2(self.c2(x, ei)))
            x = F.dropout(x, p=self.drop, training=self.training)
            x = F.relu(self.c3(x, ei))
            return self.head(x, d.batch, d.x.view(-1, self.n))

    class GATp(Base):

        def __init__(self, n_sensors, hidden=128, heads=4, **kw):
            super().__init__(n_sensors, hidden, **kw)
            h = hidden // heads
            self.c1 = GATConv(hidden, h, heads=heads, dropout=0.3)
            self.c2 = GATConv(h * heads, h, heads=heads, dropout=0.3)
            self.c3 = GATConv(h * heads, hidden, heads=1, dropout=0.3, concat=False)

        def forward(self, d):
            x, ei = (self.stem(d), d.edge_index)
            x = F.elu(self.bn1(self.c1(x, ei)))
            x = F.dropout(x, p=self.drop, training=self.training)
            x = F.elu(self.bn2(self.c2(x, ei)))
            x = F.dropout(x, p=self.drop, training=self.training)
            x = F.elu(self.c3(x, ei))
            return self.head(x, d.batch, d.x.view(-1, self.n))

    class SAGEp(Base):

        def __init__(self, n_sensors, hidden=128, **kw):
            super().__init__(n_sensors, hidden, **kw)
            self.c1, self.c2, self.c3 = (SAGEConv(hidden, hidden), SAGEConv(hidden, hidden), SAGEConv(hidden, hidden))

        def forward(self, d):
            x, ei = (self.stem(d), d.edge_index)
            x = F.relu(self.bn1(self.c1(x, ei)))
            x = F.dropout(x, p=self.drop, training=self.training)
            x = F.relu(self.bn2(self.c2(x, ei)))
            x = F.dropout(x, p=self.drop, training=self.training)
            x = F.relu(self.c3(x, ei))
            return self.head(x, d.batch, d.x.view(-1, self.n))
    return {'GCN+': GCNp, 'GAT+': GATp, 'GraphSAGE+': SAGEp}
