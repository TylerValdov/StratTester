import { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, LineData } from 'lightweight-charts';

interface EquityPoint {
  time: string;
  value: number;
}

interface BenchmarkChartProps {
  strategyData: EquityPoint[];
  buyHoldData?: EquityPoint[];
  spyData?: EquityPoint[];
  height?: number;
}

export default function BenchmarkChart({
  strategyData,
  buyHoldData,
  spyData,
  height = 400,
}: BenchmarkChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height,
      layout: {
        background: { type: ColorType.Solid, color: 'white' },
        textColor: '#333',
      },
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      rightPriceScale: {
        borderColor: '#ddd',
      },
      timeScale: {
        borderColor: '#ddd',
        timeVisible: true,
      },
    });

    chartRef.current = chart;

    // Add strategy equity line (blue)
    const strategySeries = chart.addLineSeries({
      color: '#2563eb',
      lineWidth: 2,
      title: 'Strategy',
    });
    strategySeries.setData(strategyData as LineData[]);

    // Add buy-and-hold benchmark (orange)
    if (buyHoldData && buyHoldData.length > 0) {
      const buyHoldSeries = chart.addLineSeries({
        color: '#f97316',
        lineWidth: 2,
        title: 'Buy & Hold',
      });
      buyHoldSeries.setData(buyHoldData as LineData[]);
    }

    // Add SPY benchmark (green)
    if (spyData && spyData.length > 0) {
      const spySeries = chart.addLineSeries({
        color: '#10b981',
        lineWidth: 2,
        title: 'SPY',
      });
      spySeries.setData(spyData as LineData[]);
    }

    chart.timeScale().fitContent();

    // Handle window resize
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
  }, [strategyData, buyHoldData, spyData, height]);

  return <div ref={chartContainerRef} className="w-full" />;
}
