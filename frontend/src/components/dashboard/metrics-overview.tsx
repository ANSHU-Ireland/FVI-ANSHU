'use client';

import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { apiClient } from '@/lib/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { formatNumber, formatPercentage } from '@/lib/utils';
import { TrendingUp, TrendingDown } from 'lucide-react';

export function MetricsOverview() {
  const { data: metrics, isLoading } = useQuery({
    queryKey: ['metrics'],
    queryFn: () => apiClient.getMetrics({ limit: 20 }),
    refetchInterval: 30000,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Metrics Overview</CardTitle>
          <CardDescription>Loading metrics data...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] flex items-center justify-center">
            <div className="text-sm text-muted-foreground">Loading...</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!metrics?.data?.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Metrics Overview</CardTitle>
          <CardDescription>No metrics data available</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px] flex items-center justify-center">
            <div className="text-sm text-muted-foreground">No data available</div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Group metrics by mine for visualization
  const mineMetrics = metrics.data.reduce((acc, metric) => {
    if (!acc[metric.mine_id]) {
      acc[metric.mine_id] = {
        mine_id: metric.mine_id,
        total_value: 0,
        count: 0,
        avg_trust: 0,
        avg_quality: 0,
      };
    }
    acc[metric.mine_id].total_value += metric.value;
    acc[metric.mine_id].count += 1;
    acc[metric.mine_id].avg_trust += metric.trust_score;
    acc[metric.mine_id].avg_quality += metric.data_quality_score;
    return acc;
  }, {} as Record<string, any>);

  // Calculate averages and prepare chart data
  const chartData = Object.values(mineMetrics).map((mine: any) => ({
    mine_id: mine.mine_id.slice(0, 8), // Truncate for display
    total_value: mine.total_value,
    avg_trust: (mine.avg_trust / mine.count) * 100,
    avg_quality: (mine.avg_quality / mine.count) * 100,
  }));

  // Time series data for trend analysis
  const timeSeriesData = metrics.data
    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
    .slice(-10)
    .map((metric) => ({
      timestamp: new Date(metric.timestamp).toLocaleTimeString(),
      value: metric.value,
      trust_score: metric.trust_score * 100,
      quality_score: metric.data_quality_score * 100,
    }));

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5" />
          Metrics Overview
        </CardTitle>
        <CardDescription>
          Real-time metrics across all mining operations
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Mine Performance Chart */}
          <div>
            <h4 className="text-sm font-medium mb-3">Performance by Mine</h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="mine_id" />
                <YAxis />
                <Tooltip 
                  formatter={(value, name) => {
                    if (name === 'total_value') return [formatNumber(value as number), 'Total Value'];
                    if (name === 'avg_trust') return [`${(value as number).toFixed(1)}%`, 'Trust Score'];
                    if (name === 'avg_quality') return [`${(value as number).toFixed(1)}%`, 'Quality Score'];
                    return [value, name];
                  }}
                />
                <Bar dataKey="total_value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Trend Analysis */}
          <div>
            <h4 className="text-sm font-medium mb-3">Recent Trends</h4>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={timeSeriesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip 
                  formatter={(value, name) => {
                    if (name === 'value') return [formatNumber(value as number), 'Value'];
                    if (name === 'trust_score') return [`${(value as number).toFixed(1)}%`, 'Trust Score'];
                    if (name === 'quality_score') return [`${(value as number).toFixed(1)}%`, 'Quality Score'];
                    return [value, name];
                  }}
                />
                <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} />
                <Line type="monotone" dataKey="trust_score" stroke="#82ca9d" strokeWidth={2} />
                <Line type="monotone" dataKey="quality_score" stroke="#ffc658" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {formatNumber(metrics.data.length)}
              </div>
              <div className="text-sm text-muted-foreground">Total Metrics</div>
            </div>
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {formatPercentage(
                  metrics.data.reduce((acc, m) => acc + m.trust_score, 0) / metrics.data.length
                )}
              </div>
              <div className="text-sm text-muted-foreground">Avg Trust</div>
            </div>
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="text-2xl font-bold text-orange-600">
                {formatPercentage(
                  metrics.data.reduce((acc, m) => acc + m.data_quality_score, 0) / metrics.data.length
                )}
              </div>
              <div className="text-sm text-muted-foreground">Avg Quality</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
