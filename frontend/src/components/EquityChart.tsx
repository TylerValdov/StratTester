import { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi } from 'lightweight-charts';

interface EquityPoint {
  time: string;
  value: number;
}

interface EquityChartProps {
  equityData: EquityPoint[];
  height?: number;
}

export default function EquityChart({ equityData, height = 300 }: EquityChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Area'> | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#ffffff' },
        textColor: '#333',
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#cccccc',
      },
      timeScale: {
        borderColor: '#cccccc',
        timeVisible: true,
      },
    });

    chartRef.current = chart;

    // Add area series for equity curve
    const areaSeries = chart.addAreaSeries({
      topColor: 'rgba(33, 150, 243, 0.4)',
      bottomColor: 'rgba(33, 150, 243, 0.0)',
      lineColor: 'rgba(33, 150, 243, 1)',
      lineWidth: 2,
    });

    seriesRef.current = areaSeries;

    // Set equity data
    if (equityData.length > 0) {
      areaSeries.setData(equityData);
    }

    // Fit content
    chart.timeScale().fitContent();

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [equityData, height]);

  return <div ref={chartContainerRef} className="w-full" />;
}
