'use client';

import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { apiClient, type PredictionResult } from '@/lib/api';
import { formatDate, formatNumber } from '@/lib/utils';
import { Brain, TrendingUp, Target, Clock, Zap } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ScatterChart, Scatter } from 'recharts';

interface RecentPredictionsProps {
  showAll?: boolean;
}

export function RecentPredictions({ showAll = false }: RecentPredictionsProps) {
  // Mock data for demonstration since we don't have actual prediction endpoints
  const mockPredictions: PredictionResult[] = [
    {
      mine_id: 'mine_001',
      prediction: 0.85,
      confidence: 0.92,
      explainability: {
        'production_rate': 0.35,
        'equipment_health': 0.25,
        'environmental_factors': 0.20,
        'maintenance_schedule': 0.15,
        'weather_conditions': 0.05,
      },
      model_version: 'v2.1.0',
      timestamp: new Date().toISOString(),
    },
    {
      mine_id: 'mine_002',
      prediction: 0.72,
      confidence: 0.88,
      explainability: {
        'production_rate': 0.40,
        'equipment_health': 0.30,
        'environmental_factors': 0.15,
        'maintenance_schedule': 0.10,
        'weather_conditions': 0.05,
      },
      model_version: 'v2.1.0',
      timestamp: new Date(Date.now() - 300000).toISOString(),
    },
    {
      mine_id: 'mine_003',
      prediction: 0.91,
      confidence: 0.95,
      explainability: {
        'production_rate': 0.45,
        'equipment_health': 0.20,
        'environmental_factors': 0.18,
        'maintenance_schedule': 0.12,
        'weather_conditions': 0.05,
      },
      model_version: 'v2.1.0',
      timestamp: new Date(Date.now() - 600000).toISOString(),
    },
    {
      mine_id: 'mine_004',
      prediction: 0.68,
      confidence: 0.83,
      explainability: {
        'production_rate': 0.38,
        'equipment_health': 0.32,
        'environmental_factors': 0.15,
        'maintenance_schedule': 0.10,
        'weather_conditions': 0.05,
      },
      model_version: 'v2.1.0',
      timestamp: new Date(Date.now() - 900000).toISOString(),
    },
  ];

  const predictions = showAll ? mockPredictions : mockPredictions.slice(0, 3);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.8) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.9) return 'bg-green-100 text-green-800';
    if (confidence >= 0.8) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getPredictionColor = (prediction: number) => {
    if (prediction >= 0.8) return 'text-green-600';
    if (prediction >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  // Prepare data for visualization
  const chartData = predictions.map(pred => ({
    mine_id: pred.mine_id.slice(-3),
    prediction: pred.prediction * 100,
    confidence: pred.confidence * 100,
  }));

  const explainabilityData = predictions[0] ? Object.entries(predictions[0].explainability).map(([feature, importance]) => ({
    feature: feature.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
    importance: importance * 100,
  })) : [];

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Predictions</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{predictions.length}</div>
            <p className="text-xs text-muted-foreground">
              Recent predictions
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Confidence</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getConfidenceColor(
              predictions.reduce((acc, p) => acc + p.confidence, 0) / predictions.length
            )}`}>
              {formatNumber(predictions.reduce((acc, p) => acc + p.confidence, 0) / predictions.length * 100, 1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              Model confidence
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Prediction</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${getPredictionColor(
              predictions.reduce((acc, p) => acc + p.prediction, 0) / predictions.length
            )}`}>
              {formatNumber(predictions.reduce((acc, p) => acc + p.prediction, 0) / predictions.length * 100, 1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              Performance score
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Version</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {predictions[0]?.model_version || 'N/A'}
            </div>
            <p className="text-xs text-muted-foreground">
              Current model
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Predictions Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            Prediction vs Confidence
          </CardTitle>
          <CardDescription>
            Model predictions with confidence intervals
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="mine_id" />
                <YAxis domain={[0, 100]} />
                <Tooltip 
                  formatter={(value, name) => [
                    `${value}%`, 
                    name === 'prediction' ? 'Prediction' : 'Confidence'
                  ]}
                />
                <Bar dataKey="prediction" fill="#8884d8" />
                <Bar dataKey="confidence" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Feature Importance */}
      {explainabilityData.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Feature Importance
            </CardTitle>
            <CardDescription>
              SHAP values for the most recent prediction
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={explainabilityData} layout="horizontal">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 50]} />
                  <YAxis type="category" dataKey="feature" width={120} />
                  <Tooltip 
                    formatter={(value) => [`${value}%`, 'Importance']}
                  />
                  <Bar dataKey="importance" fill="#ffc658" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Predictions List */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Recent Predictions
          </CardTitle>
          <CardDescription>
            Latest ML predictions with explainability
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {predictions.map((prediction, index) => (
              <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <Brain className="h-4 w-4 text-muted-foreground" />
                    <div>
                      <div className="font-medium">{prediction.mine_id}</div>
                      <div className="text-sm text-muted-foreground">
                        {formatDate(prediction.timestamp)}
                      </div>
                    </div>
                  </div>
                  <Badge variant="outline" className="text-xs">
                    {prediction.model_version}
                  </Badge>
                </div>
                <div className="flex items-center gap-6">
                  <div className="text-right">
                    <div className={`text-lg font-bold ${getPredictionColor(prediction.prediction)}`}>
                      {formatNumber(prediction.prediction * 100, 1)}%
                    </div>
                    <div className="text-xs text-muted-foreground">Prediction</div>
                  </div>
                  <div className="text-right">
                    <div className={`text-lg font-bold ${getConfidenceColor(prediction.confidence)}`}>
                      {formatNumber(prediction.confidence * 100, 1)}%
                    </div>
                    <div className="text-xs text-muted-foreground">Confidence</div>
                  </div>
                  <Badge className={getConfidenceBadge(prediction.confidence)}>
                    {prediction.confidence >= 0.9 ? 'High' : prediction.confidence >= 0.8 ? 'Medium' : 'Low'}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
