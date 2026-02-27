import {
  type CatalogItem,
  photographyLabels,
  locationLabels,
} from "../data/products";

interface ProductCardProps {
  item: CatalogItem;
  onSelect: (item: CatalogItem) => void;
  disabled: boolean;
  index: number;
}

export function ProductCard({ item, onSelect, disabled, index }: ProductCardProps): JSX.Element {
  const poseClass = item.modelPhotography === 1 ? "pose-front" : "pose-profile";

  return (
    <button
      type="button"
      className="photo-card"
      onClick={() => onSelect(item)}
      disabled={disabled}
      style={{ animationDelay: `${index * 45}ms` }}
      aria-label={`View model ${item.modelCode} at ${locationLabels[item.location]} (${photographyLabels[item.modelPhotography]})`}
    >
      <div className="mock-photo" data-location={item.location}>
        <span className="mock-location">{locationLabels[item.location]}</span>
        <span className="mock-pose">{photographyLabels[item.modelPhotography]}</span>
        <span className={`model-shape ${poseClass}`} aria-hidden="true" />
      </div>

      <div className="photo-meta">
        <span className="photo-badge">{item.badge}</span>
        <h3>{item.title}</h3>
        <p>
          Model {item.modelCode} • {item.colourName}
        </p>
        <strong>${item.price.toFixed(0)}</strong>
      </div>
    </button>
  );
}
