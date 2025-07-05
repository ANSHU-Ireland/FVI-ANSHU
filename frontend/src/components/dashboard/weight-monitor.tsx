'use client';

import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { apiClient, type WeightUpdate } from '@/lib/api';
import { formatDate, formatNumber } from '@/lib/utils';
import { TrendingUp, TrendingDown, Settings, Clock, BarChart3 } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar } from 'recharts';

export function WeightMonitor() {
  const { data: currentWeights, isLoading: weightsLoading } = useQuery({
    queryKey: ['current-weights'],
    queryFn: () => apiClient.getCurrentWeights(),
    refetchInterval: 60000, // Refetch every minute
  });

  const { data: weightHistory, isLoading: historyLoading } = useQuery({
    queryKey: ['weight-history'],
    queryFn: () => apiClient.getWeightHistory(50),
    refetchInterval: 60000,
  });

  const isLoading = weightsLoading || historyLoading;

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Dynamic Weight Monitor</CardTitle>
          <CardDescription>Loading weight data...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center">
            <div className="text-sm text-muted-foreground">Loading...</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!currentWeights?.data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Dynamic Weight Monitor</CardTitle>
          <CardDescription>No weight data available</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[400px] flex items-center justify-center">
            <div className="text-sm text-muted-foreground">No data available</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Prepare current weights data for visualization
  const currentWeightsData = Object.entries(currentWeights.data).map(([feature, weight]) => ({
    feature,
    weight: weight as number,
    formatted_weight: formatNumber(weight as number, 4),
  }));

  // Prepare weight history data for trend analysis
  const weightHistoryData = weightHistory?.data?.map((update: WeightUpdate, index: number) => ({
    index,
    timestamp: new Date(update.timestamp).toLocaleTimeString(),
    feature: update.feature_name,
    old_weight: update.old_weight,
    new_weight: update.new_weight,
    change: update.new_weight - update.old_weight,
    information_gain: update.information_gain,
  })) || [];

  // Calculate recent changes
  const recentChanges = weightHistory?.data?.slice(0, 10) || [];
  const totalFeatures = Object.keys(currentWeights.data).length;
  const recentUpdates = recentChanges.length;

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Features</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalFeatures}</div>
            <p className="text-xs text-muted-foreground">
              Active feature weights
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Recent Updates</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{recentUpdates}</div>
            <p className="text-xs text-muted-foreground">
              In the last period
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Weight</CardTitle>
            <Settings className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatNumber(
                Object.values(currentWeights.data).reduce((acc, w) => acc + (w as number), 0) / totalFeatures,
                4
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Across all features
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Info Gain</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatNumber(
                recentChanges.reduce((acc, change) => acc + change.information_gain, 0),
                4
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              Recent period
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Current Weights */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Current Feature Weights
          </CardTitle>
          <CardDescription>
            Real-time feature importance weights from the dynamic weight engine
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={currentWeightsData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="feature" 
                  angle={-45}
                  textAnchor="end"
                  height={100}
                />
                <YAxis />
                <Tooltip 
                  formatter={(value) => [formatNumber(value as number, 4), 'Weight']}
                />
                <Bar dataKey="weight" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Weight History */}
      {weightHistory?.data?.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Weight Change History
            </CardTitle>
            <CardDescription>
              Recent weight updates from the information gain engine
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {/* Recent Changes List */}
              <div className="max-h-[300px] overflow-y-auto space-y-2">
                {recentChanges.map((change: WeightUpdate, index: number) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-1">
                        {change.new_weight > change.old_weight ? (
                          <TrendingUp className="h-4 w-4 text-green-600" />
                        ) : (
                          <TrendingDown className="h-4 w-4 text-red-600" />
                        )}
                        <span className="font-medium">{change.feature_name}</span>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {formatDate(change.timestamp)}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <div className="text-muted-foreground">
                        {formatNumber(change.old_weight, 4)} → {formatNumber(change.new_weight, 4)}
                      </div>
                      <div className={`font-medium ${
                        change.new_weight > change.old_weight ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {change.new_weight > change.old_weight ? '+' : ''}
                        {formatNumber(change.new_weight - change.old_weight, 4)}
                      </div>
                      <div className="text-blue-600 font-medium">
                        IG: {formatNumber(change.information_gain, 4)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Information Gain Summary */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t">
                <div className="text-center p-3 bg-muted rounded-lg">
                  <div className="text-lg font-bold text-green-600">
                    {recentChanges.filter(c => c.new_weight > c.old_weight).length}
                  </div>
                  <div className="text-sm text-muted-foreground">Weight Increases</div>
                </div>
                <div className="text-center p-3 bg-muted rounded-lg">
                  <div className="text-lg font-bold text-red-600">
                    {recentChanges.filter(c => c.new_weight < c.old_weight).length}
                  </div>
                  <div className="text-sm text-muted-foreground">Weight Decreases</div>
                </div>
                <div className="text-center p-3 bg-muted rounded-lg">
                  <div className="text-lg font-bold text-blue-600">
                    {formatNumber(
                      recentChanges.reduce((acc, c) => acc + Math.abs(c.new_weight - c.old_weight), 0),
                      4
                    )}
                  </div>
                  <div className="text-sm text-muted-foreground">Total Change</div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
